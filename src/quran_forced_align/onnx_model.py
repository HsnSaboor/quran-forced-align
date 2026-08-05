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


def make_onnx_session(model_path):
    """Deterministic onnxruntime session: single-threaded, sequential
    execution. Rules out any thread-race nondeterminism in parallelized
    reduction ops (e.g. matmul/layernorm) across repeated runs -- required
    since the user explicitly needs bit-identical output on every run.
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])


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
        elif inp.name == "embed_states":
            dims = [1 if d == "N" else d for d in inp.shape]
            state[inp.name] = np.zeros(tuple(dims), dtype=np.float32)
        else:
            dims = [1 if d == "N" else d for d in inp.shape]
            state[inp.name] = np.zeros(tuple(dims), dtype=np.float32)
    return state


def run_streaming_log_probs(sess, feats):
    """Feed 80-dim fbank features through the streaming Zipformer2-CTC ONNX
    graph chunk-by-chunk, threading the 96 cache tensors + embed_states +
    processed_lens between calls, and concatenate the per-chunk log_probs
    outputs into one [T_total, 251] matrix for the whole utterance.

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

    state = _zero_state_for_inputs(inputs)
    all_log_probs = []
    ptr = 0
    for _ in range(n_chunks):
        chunk = feats[ptr:ptr + segment][None, :, :].astype(np.float32)
        feed = {"x": chunk}
        feed.update(state)
        out = sess.run(None, feed)
        d = dict(zip(out_names, out))
        all_log_probs.append(d["log_probs"][0])
        state = {inp.name: d["new_" + inp.name] for inp in inputs if inp.name != "x"}
        ptr += offset

    log_probs = np.concatenate(all_log_probs, axis=0).astype(np.float64)
    frames_per_chunk = all_log_probs[0].shape[0]
    subsample_factor = offset / frames_per_chunk  # empirically 48/12 = 4
    seconds_per_output_frame = FRAME_SHIFT_SEC * subsample_factor
    return log_probs, seconds_per_output_frame
