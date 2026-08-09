"""Raw-ONNX streaming Zipformer2-CTC chunk loop -> full [T,251] log-probs.

Bypasses sherpa_onnx's Python API entirely (deliberately not a dependency of
this package -- see the package docstring for why the full per-frame
log-probability matrix this pipeline needs is not obtainable via
sherpa_onnx's Python API at all).
"""
import math

import numpy as np
import onnxruntime as ort

from .constants import FRAME_SHIFT_SEC


def make_onnx_session(model_path, providers=("CPUExecutionProvider",)):
    """Deterministic onnxruntime session: single-threaded, sequential
    execution. Rules out any thread-race nondeterminism in parallelized
    reduction ops (e.g. matmul/layernorm) across repeated runs -- required
    since the user explicitly needs bit-identical output on every run.

    `providers` defaults to CPU-only (this function's original,
    unconditional behaviour, unchanged for every existing caller). Passing
    `["CUDAExecutionProvider", "CPUExecutionProvider"]` (see
    `engines.cuda.CUDAEngine`) runs the same single-threaded/sequential
    settings on GPU instead -- verified empirically (against a real T4 GPU
    session) to still produce bit-identical repeated-run output, since
    onnxruntime's CUDA EP has no thread-count knob of its own to pin here
    (GPU kernels are launched from ORT's single CPU-side control thread
    regardless of `intra_op_num_threads`; determinism there comes from the
    CUDA kernels themselves always reducing in the same fixed order for a
    fixed input, not from a CPU thread-count setting).
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(model_path, sess_options=so, providers=list(providers))


def _zero_state_for_inputs(sess_inputs):
    """Initialize all 96 cache tensors + embed_states + processed_lens to
    zeros, with shapes read from the ACTUAL loaded ONNX graph's
    get_inputs() (not hardcoded guesses) -- the 'N' (batch) dim is always 1
    for our single-utterance-at-a-time use, other dims come straight from
    the graph.
    """
    state = {}
    for inp in sess_inputs:
        if inp.name == "x":
            continue
        if inp.name == "processed_lens":
            state[inp.name] = np.zeros((1,), dtype=np.int64)
        else:
            dims = [1 if d == "N" else d for d in inp.shape]
            state[inp.name] = np.zeros(tuple(dims), dtype=np.float32)
    return state


def run_streaming_log_probs(sess, feats, output_dtype=np.float64):
    """Feed 80-dim fbank features through the streaming Zipformer2-CTC ONNX
    graph chunk-by-chunk, threading the 96 cache tensors + embed_states +
    processed_lens between calls, and concatenate the per-chunk log_probs
    outputs into one [T_total, 251] matrix for the whole utterance.

    `output_dtype` defaults to float64 -- the CPU engine's numpy Viterbi
    DP (`viterbi/dp.py`) accumulates in float64 for its own numerical
    reasons, and this function's original (pre-multi-engine) behavior was
    to always upcast here so every existing caller kept getting float64
    with zero code change. The CUDA engine (`engines/cuda.py`) passes
    `output_dtype=np.float32` explicitly: it immediately re-casts its
    input tensor to float32 anyway for `torchaudio.functional.forced_align`
    (which only accepts float32 log-probabilities), so upcasting to
    float64 here first would only add an avoidable full-matrix
    float32->float64->float32 round trip -- confirmed as a real,
    avoidable ~2x memory/CPU waste at Al-Baqarah scale in code review, with
    zero benefit since the float64 precision is discarded again
    immediately.

    Chunk geometry is read from the model's own ONNX metadata
    (decode_chunk_len, T) rather than hardcoded, confirmed empirically to
    match: each call consumes `segment` (T=61) raw feature frames but the
    read pointer only advances by `offset` (decode_chunk_len=48) frames per
    call (the T-offset=13 frame overlap supplies left-context, matching
    icefall's export-onnx-streaming.py convention of pad_length=7+2*3=13).
    Each call emits a FIXED number of output frames (12, confirmed
    empirically = one 4x-subsampled 48-frame chunk), independent of chunk
    index.

    The last chunk is padded with trailing zero (silence) feature frames if
    needed to reach a full `segment` length -- same idea as the old
    build_surah_srt.py's decode_full() end-of-stream flush, just applied at
    the raw-feature-frame level instead of via sherpa's high-level stream
    API.
    """
    meta = sess.get_modelmeta().custom_metadata_map
    offset = int(meta.get("decode_chunk_len", 48))
    segment = int(meta.get("T", 61))

    inputs = sess.get_inputs()
    out_names = [o.name for o in sess.get_outputs()]
    # Precompute the static input-name <-> output-index correspondence once
    # (instead of rebuilding a "new_" + name string and a fresh
    # dict(zip(...)) every chunk -- for a huge surah this loop runs ~14,600
    # times, so ~97 string concats + dict builds per call adds up to over a
    # million redundant small allocations across the whole file). Every
    # cache tensor's output name is deterministically "new_" + its input
    # name and the set of names never changes between chunks, so this
    # mapping is computed exactly once, outside the loop, with no change to
    # which values end up feeding into which input on any given call.
    log_probs_out_idx = out_names.index("log_probs")
    state_names = [inp.name for inp in inputs if inp.name != "x"]
    state_out_idx = [out_names.index("new_" + name) for name in state_names]

    T_raw, feat_dim = feats.shape
    if T_raw <= segment:
        n_chunks = 1
    else:
        n_chunks = 1 + math.ceil((T_raw - segment) / offset)
    total_len_needed = segment + (n_chunks - 1) * offset
    pad_needed = max(0, total_len_needed - T_raw)
    if pad_needed > 0:
        feats = np.concatenate(
            [feats, np.zeros((pad_needed, feat_dim), dtype=np.float32)], axis=0
        )

    # `feed` is reused across every chunk (same fixed set of keys every
    # call: "x" plus every cache-tensor input name) -- only the VALUES
    # change per iteration, so there's no need to allocate a fresh dict via
    # feed.update(state) each time.
    feed = {"x": None}
    feed.update(_zero_state_for_inputs(inputs))

    log_probs = None  # preallocated once the first chunk reveals frames_per_chunk
    ptr = 0
    for chunk_idx in range(n_chunks):
        # copy=False: the slice is already float32 (feats is always
        # float32 by construction -- see compute_fbank_features/the
        # padding above), so this only forces a copy when one is actually
        # needed to materialize the [None, :, :] view as contiguous input
        # for onnxruntime, never as an unconditional dtype-cast copy.
        chunk = feats[ptr:ptr + segment][None, :, :].astype(np.float32, copy=False)
        feed["x"] = chunk
        out = sess.run(None, feed)

        chunk_log_probs = out[log_probs_out_idx][0]
        if log_probs is None:
            frames_per_chunk = chunk_log_probs.shape[0]
            log_probs = np.empty((n_chunks * frames_per_chunk, chunk_log_probs.shape[1]),
                                  dtype=np.float32)
        # Defensive check: the preallocation above and the row_start/
        # frames_per_chunk indexing below both assume every chunk emits
        # EXACTLY frames_per_chunk (fixed, from chunk 0) output frames. If
        # the model ever violates that invariant (e.g. a different chunk
        # emits a different number of frames), fail loudly here instead of
        # silently misaligning or truncating output rows.
        assert chunk_log_probs.shape[0] == frames_per_chunk, (
            f"chunk {chunk_idx} emitted {chunk_log_probs.shape[0]} frames, "
            f"expected fixed frames_per_chunk={frames_per_chunk} (from chunk 0)"
        )
        row_start = chunk_idx * frames_per_chunk
        log_probs[row_start:row_start + frames_per_chunk] = chunk_log_probs

        for name, out_idx in zip(state_names, state_out_idx):
            feed[name] = out[out_idx]
        ptr += offset

    log_probs = log_probs.astype(output_dtype)
    subsample_factor = offset / frames_per_chunk  # empirically 48/12 = 4
    seconds_per_output_frame = FRAME_SHIFT_SEC * subsample_factor
    return log_probs, seconds_per_output_frame
