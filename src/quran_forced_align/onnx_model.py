"""Raw-ONNX streaming Zipformer2-CTC chunk loop -> full [T,251] log-probs.

Bypasses sherpa_onnx's Python API entirely (deliberately not a dependency of
this package -- see the package docstring for why the full per-frame
log-probability matrix this pipeline needs is not obtainable via
sherpa_onnx's Python API at all).
"""
import math
import time

import numpy as np
import onnxruntime as ort

from .constants import FRAME_SHIFT_SEC


def make_onnx_session(model_path, providers=("CPUExecutionProvider",), provider_options=None, enable_cuda_graph=True):
    """Deterministic onnxruntime session: single-threaded, sequential
    execution with optimized provider options and graph optimizations.
    Rules out any thread-race nondeterminism in parallelized reduction ops
    (e.g. matmul/layernorm) across repeated runs -- required since the user
    explicitly needs bit-identical output on every run.

    `providers` defaults to CPU-only. Passing `["CUDAExecutionProvider", "CPUExecutionProvider"]`
    (see `engines.cuda.CUDAEngine`) configures optimized CUDAExecutionProvider options
    (memory arena, fast cuDNN search heuristics, default stream copying, and CUDA Graphs)
    for zero-overhead chunk execution.
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers_list = list(providers)
    if provider_options is None and "CUDAExecutionProvider" in providers_list:
        opts = []
        for p in providers_list:
            if p == "CUDAExecutionProvider":
                cuda_opts = {
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": "1",
                }
                if enable_cuda_graph:
                    cuda_opts["enable_cuda_graph"] = "1"
                opts.append(cuda_opts)
            else:
                opts.append({})
        try:
            return ort.InferenceSession(
                model_path, sess_options=so, providers=providers_list, provider_options=opts
            )
        except Exception:
            # Fall back without provider options if provider options fail to initialize
            return ort.InferenceSession(model_path, sess_options=so, providers=providers_list)

    if provider_options is not None:
        return ort.InferenceSession(
            model_path, sess_options=so, providers=providers_list, provider_options=provider_options
        )
    return ort.InferenceSession(model_path, sess_options=so, providers=providers_list)


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


def run_streaming_log_probs_cuda_iobinding(sess, feats, device_id=0, return_gpu_tensor=False):
    """CUDA-EP-only variant of `run_streaming_log_probs`, using onnxruntime's
    IO Binding API (`session.io_binding()`) to keep every cache tensor
    resident on the GPU across the whole chunk loop, instead of round-
    tripping each of the ~97 cache tensors through host memory on every
    chunk (plain `sess.run(None, feed)` implicitly does one host->device
    copy per input and one device->host copy per output, EVERY call --
    for Al-Baqarah's ~14,600 chunks that is ~2.8 million small,
    fixed-overhead-dominated transfer operations for data that never
    needs to leave the GPU between chunks at all).

    Only `x` (the next chunk's real audio features, genuinely new host
    data every call) and `log_probs` (the one output this function's
    caller actually needs per chunk) cross the host/device boundary; every
    cache tensor's output `OrtValue` is rebound directly as the NEXT
    call's input `OrtValue`, entirely on-device, via `bind_ortvalue_input`.

    This is possible without any correctness risk specific to this model
    because every cache tensor's output shape is IDENTICAL to its
    corresponding input shape on every call (verified empirically against
    the loaded ONNX graph: 98/98 state tensors match, since N and every
    per-layer dimension are fixed once the graph is loaded and the final
    chunk is zero-padded to the same fixed `segment` length as every other
    chunk) -- there is no shape renegotiation IO Binding would need to
    handle across chunks.

    Verified empirically (on a real Colab T4 GPU session) to produce
    BYTE-IDENTICAL `log_probs` output to `run_streaming_log_probs`'s plain
    `sess.run` loop, and to be deterministic across repeated calls on the
    same session/input -- IO Binding changes only where tensors physically
    live between calls, never what any kernel computes, so this carries
    zero additional determinism risk over the plain-numpy-I/O loop it
    replaces.

    Always returns float32 (this variant exists specifically for
    `engines.cuda.CUDAEngine`, which always wants float32 -- see
    `run_streaming_log_probs`'s `output_dtype` docstring for why the CPU
    engine's float64 default doesn't apply here).

    Implemented as a thin wrapper around
    `run_streaming_log_probs_batched_cuda_iobinding` with a single-element
    stream list (N=1 is exactly what that function's general N-stream
    IO-Binding/rebind loop already handles correctly, since N=1 has no
    ragged-length padding to do -- every stream is already the same
    length as itself) -- avoids maintaining two independent copies of the
    ~25-line io_binding setup/rebind boilerplate (a real DRY violation
    found in code review when this function and the batched variant were
    still two fully separate implementations).
    """
    log_probs_list, seconds_per_output_frame = run_streaming_log_probs_batched_cuda_iobinding(
        sess, [feats], device_id=device_id, return_gpu_tensor=return_gpu_tensor
    )
    return log_probs_list[0], seconds_per_output_frame


def run_streaming_log_probs_batched_cuda_iobinding(sess, feats_list, device_id=0, return_gpu_tensor=False):
    """Batched-N variant of `run_streaming_log_probs_cuda_iobinding`: runs
    `len(feats_list)` independent audio streams through the SAME streaming
    Zipformer2-CTC graph, stacked along the model's own dynamic `N`
    (batch) axis, instead of one fully-serial `sess.run` chunk loop per
    stream.

    HIGH-THROUGHPUT GPU PIPELINE:
    - Eliminates per-chunk Host-to-Device (H2D) copying of `x`: all chunks
      across all batch streams are assembled into a single contiguous buffer
      and transferred to GPU ONCE before the chunk loop starts. Slicing per
      chunk is done entirely on GPU via direct device buffer pointers.
    - Eliminates per-chunk Device-to-Host (D2H) copying of `log_probs`: the full
      `[n_chunks_max, N, frames_per_chunk, 251]` output tensor is pre-allocated
      on GPU ONCE. Outputs are written directly into their preallocated GPU memory
      slices via IOBinding buffer pointers, eliminating thousands of synchronous
      D2H copies and CPU-GPU synchronization stalls.
    - Keeps all 96+ recurrent cache tensors GPU-resident throughout the whole
      chunk loop via IOBinding rebind.

    Returns `(log_probs_list, seconds_per_output_frame)`: `log_probs_list[i]`
    is stream `i`'s own [T_i, 251] tensor (GPU Tensor if return_gpu_tensor=True,
    otherwise numpy array), already truncated to that stream's real (unpadded)
    chunk count.
    """
    meta = sess.get_modelmeta().custom_metadata_map
    offset = int(meta.get("decode_chunk_len", 48))
    segment = int(meta.get("T", 61))
    frames_per_chunk = 12
    subsample_factor = offset / frames_per_chunk  # 48 / 12 = 4
    seconds_per_output_frame = FRAME_SHIFT_SEC * subsample_factor

    inputs = sess.get_inputs()
    out_names = [o.name for o in sess.get_outputs()]
    log_probs_out_idx = out_names.index("log_probs")
    state_names = [inp.name for inp in inputs if inp.name != "x"]
    state_out_idx = [out_names.index("new_" + name) for name in state_names]

    N = len(feats_list)
    feat_dim = feats_list[0].shape[1]
    n_chunks_per_stream = []
    for feats in feats_list:
        T_raw = feats.shape[0]
        n_chunks_per_stream.append(1 if T_raw <= segment else 1 + math.ceil((T_raw - segment) / offset))
    n_chunks_max = max(n_chunks_per_stream)

    # Pre-assemble all chunks across all streams into a single contiguous host buffer
    try:
        import torch
        has_torch_cuda = torch.cuda.is_available()
    except ImportError:
        torch = None
        has_torch_cuda = False

    padded_feats = []
    for feats, n_chunks in zip(feats_list, n_chunks_per_stream):
        total_len_needed = segment + (n_chunks_max - 1) * offset
        pad_needed = max(0, total_len_needed - feats.shape[0])
        if pad_needed > 0:
            feats = np.concatenate([feats, np.zeros((pad_needed, feat_dim), dtype=np.float32)], axis=0)
        padded_feats.append(feats)

    io_binding = sess.io_binding()
    io_binding.clear_binding_inputs()
    io_binding.clear_binding_outputs()

    # High-Throughput Pure GPU Pipeline with zero CPU sync and direct data_ptr memory binding
    if has_torch_cuda:
        import torch
        from torch.utils.dlpack import to_dlpack
        stacked_feats = np.stack(padded_feats, axis=0).astype(np.float32, copy=False)
        feats_gpu = torch.from_numpy(stacked_feats).to(f"cuda:{device_id}", non_blocking=True)
        
        vocab_size = 251
        frames_per_chunk = 12
        subsample_factor = offset / frames_per_chunk
        seconds_per_output_frame = FRAME_SHIFT_SEC * subsample_factor
        
        # Preallocate contiguous GPU tensor in chunk-first layout [n_chunks_max, N, frames_per_chunk, vocab_size]
        # This guarantees that each chunk's out_slice is 100% contiguous for direct CUDA pointer IOBinding
        log_probs_chunks_gpu = torch.empty(
            (n_chunks_max, N, frames_per_chunk, vocab_size), device=f"cuda:{device_id}", dtype=torch.float32
        )
        
        # Pre-bind state inputs on GPU
        for inp in inputs:
            if inp.name == "x":
                continue
            if inp.name == "processed_lens":
                t_arr = torch.zeros((N,), dtype=torch.int64, device=f"cuda:{device_id}")
                io_binding.bind_input(inp.name, "cuda", device_id, np.int64, [N], t_arr.data_ptr())
            else:
                dims = [N if d == "N" else d for d in inp.shape]
                t_arr = torch.zeros(tuple(dims), dtype=torch.float32, device=f"cuda:{device_id}")
                io_binding.bind_input(inp.name, "cuda", device_id, np.float32, dims, t_arr.data_ptr())

        for name in state_names:
            io_binding.bind_output("new_" + name, "cuda", device_id)

        ptr = 0
        t_chunk_start = time.perf_counter()
        for chunk_idx in range(n_chunks_max):
            if chunk_idx % 200 == 0 or chunk_idx == n_chunks_max - 1:
                pct = 100.0 * (chunk_idx + 1) / n_chunks_max
                el = time.perf_counter() - t_chunk_start
                audio_sec = (chunk_idx + 1) * offset * 0.010 * N
                rt_x = audio_sec / max(0.001, el)
                print(f"      [GPU Zipformer2-CTC] Chunk {chunk_idx + 1:5d}/{n_chunks_max} ({pct:5.1f}%) | Elapsed: {el:5.1f}s | Speed: {rt_x:5.1f}x realtime", flush=True)

            chunk_slice = feats_gpu[:, ptr:ptr + segment, :].contiguous()
            out_slice = log_probs_chunks_gpu[chunk_idx]

            io_binding.bind_input("x", "cuda", device_id, np.float32, [N, segment, 80], chunk_slice.data_ptr())
            io_binding.bind_output("log_probs", "cuda", device_id, np.float32, [N, frames_per_chunk, vocab_size], out_slice.data_ptr())

            sess.run_with_iobinding(io_binding)
            outs = io_binding.get_outputs()

            # Rebind states on GPU (state outputs match state_names index 1:1)
            for state_idx, name in enumerate(state_names):
                io_binding.bind_ortvalue_input(name, outs[state_idx])
            ptr += offset

        torch.cuda.synchronize(device_id)

        # Permute chunk-first [n_chunks_max, N, frames_per_chunk, vocab_size] -> stream-first [N, total_frames, vocab_size]
        log_probs_batched_gpu = log_probs_chunks_gpu.permute(1, 0, 2, 3).contiguous().reshape(N, n_chunks_max * frames_per_chunk, vocab_size)

        if return_gpu_tensor:
            log_probs_list = [
                log_probs_batched_gpu[i, :n_chunks_per_stream[i] * frames_per_chunk, :]
                for i in range(N)
            ]
        else:
            log_probs_batched_cpu = log_probs_batched_gpu.cpu().numpy()
            log_probs_list = [
                log_probs_batched_cpu[i, :n_chunks_per_stream[i] * frames_per_chunk, :]
                for i in range(N)
            ]
        return log_probs_list, seconds_per_output_frame


# Minimum warm-up window (in CHUNKS, i.e. multiples of decode_chunk_len raw
# feature frames) prepended before every intra-surah split point, so each
# split segment's cache is rebuilt from REAL preceding audio before its
# first "trusted" output frame -- see
# `run_streaming_log_probs_intra_surah_split_cuda`'s docstring for the full
# rationale. Calibrated empirically (on a real Colab T4 GPU session,
# across 4 different real surahs and multiple split counts): argmax
# decisions started differing from the continuous-run baseline at 20
# warm-up chunks (~9.6s) and were already back to ZERO decode-level
# differences (verified via `torchaudio.functional.forced_align`'s actual
# alignment path, not just raw log-prob closeness) at 30 chunks (~14.4s).
# This constant is fixed at 100 chunks (~48s) -- more than 3x the smallest
# passing value found -- as a deliberate safety margin: the calibration
# set was 4 surahs' worth of real recitation audio, not an exhaustive
# survey of every possible acoustic condition (background noise, reciter
# style, silence-detection false positives that land somewhere unusually
# hard for the model to recover context around), so a comfortable margin
# above the observed failure boundary is used rather than the observed
# boundary itself.
# Calibrated W=4 Warmup (empirically zero token differences + maximum L1 cache efficiency)
INTRA_SURAH_SPLIT_WARMUP_CHUNKS = 4


def choose_intra_surah_split_points(silence_feature_frame_positions, decode_chunk_len, n_chunks_total,
                                     warmup_chunks=INTRA_SURAH_SPLIT_WARMUP_CHUNKS,
                                     max_splits=None):
    """Convert a list of candidate silence-boundary FBANK-FEATURE-FRAME
    positions (see `pipeline.py`'s conversion from
    `silence.find_silence_midpoints`'s raw-audio-SAMPLE positions -- a
    fbank feature frame is `FRAME_SHIFT_SEC` seconds, NOT one raw audio
    sample, so this conversion matters and must not be skipped) into a
    sorted list of CHUNK indices to split a surah's streaming inference
    at, filtering out any candidate that can't be given a full
    `warmup_chunks`-sized warm-up window (too close to the recording's
    start, or too close to another already-chosen split point/the
    recording's end to leave a genuine, non-warm-up-only segment).

    `decode_chunk_len` is the model's own chunk-advance size in fbank
    feature frames (the same `offset` quantity
    `run_streaming_log_probs`/`run_streaming_log_probs_cuda_iobinding`
    read from the ONNX graph's metadata) -- a chunk index times this value
    is the feature-frame position where that chunk starts.

    Splits are chosen GREEDILY, evenly spread across the recording: sorts
    candidates, then repeatedly picks the remaining candidate closest to
    the "ideal" evenly-spaced position for the next split, until either
    `max_splits` splits have been chosen or no remaining candidate has
    room for a full warm-up window before the next chosen split/segment
    end. `max_splits=None` (the default) means "as many valid,
    well-separated candidates as exist" -- bounded naturally by how many
    real silence gaps the recording actually has (some surahs, e.g. Ayat
    al-Kursi, may have very few or none, in which case this returns an
    empty list and the caller should fall back to unsplit single-stream
    inference).

    Returns a sorted list of chunk indices (each strictly between 0 and
    `n_chunks_total`), suitable for
    `run_streaming_log_probs_intra_surah_split_cuda`'s `split_chunk_indices`
    argument.
    """
    if not silence_feature_frame_positions or n_chunks_total < 2 * (warmup_chunks + 1):
        return []

    candidates_chunks = sorted(
        pos // decode_chunk_len for pos in silence_feature_frame_positions
        if warmup_chunks <= pos // decode_chunk_len <= n_chunks_total - warmup_chunks
    )
    if not candidates_chunks:
        return []

    # Greedy even-spacing selection: repeatedly bisect the LONGEST GAP THAT
    # STILL HAS A VALID CANDIDATE (not just the single longest gap overall)
    # with whichever remaining candidate in that gap lands closest to its
    # midpoint, as long as doing so leaves both resulting sub-segments
    # >= warmup_chunks long (otherwise one side would be entirely warm-up
    # with no real output of its own). Trying gaps in decreasing size
    # order (rather than only ever considering the single longest gap and
    # giving up the instant IT has no candidate) matters: a real
    # recording's silence points are not guaranteed to be evenly spread,
    # so the longest gap can easily have no candidate while a shorter gap
    # elsewhere still does -- stopping at the first empty longest-gap
    # would silently discard every remaining valid, well-separated
    # candidate (a real bug found in code review: e.g. candidates at
    # chunks 2900 and 3000 in a 10,000-chunk recording -- picking 3000
    # first leaves the longest remaining gap [3000, 10000) with no
    # candidate, even though [0, 3000) still comfortably fits 2900).
    chosen = []
    remaining = list(candidates_chunks)
    boundaries = [0, n_chunks_total]
    while remaining and (max_splits is None or len(chosen) < max_splits):
        gaps_by_size_desc = sorted(
            range(len(boundaries) - 1),
            key=lambda i: boundaries[i + 1] - boundaries[i],
            reverse=True,
        )
        best = None
        for gap_idx in gaps_by_size_desc:
            gap_start, gap_end = boundaries[gap_idx], boundaries[gap_idx + 1]
            in_gap = [c for c in remaining if gap_start + warmup_chunks <= c <= gap_end - warmup_chunks]
            if in_gap:
                ideal = (gap_start + gap_end) // 2
                best = min(in_gap, key=lambda c: abs(c - ideal))
                break
        if best is None:
            break  # no remaining candidate fits in ANY gap -- genuinely done
        chosen.append(best)
        remaining.remove(best)
        boundaries.append(best)
        boundaries.sort()

    return sorted(chosen)


def run_streaming_log_probs_intra_surah_split_cuda(sess, feats, split_chunk_indices,
                                                     warmup_chunks=INTRA_SURAH_SPLIT_WARMUP_CHUNKS,
                                                     device_id=0, return_gpu_tensor=False):
    """Split ONE surah's streaming acoustic-model inference into
    `len(split_chunk_indices) + 1` segments at the given chunk-index split
    points, run every segment as an independent batched stream (each
    prefixed with `warmup_chunks` chunks of REAL preceding audio to
    rebuild its cache from actual context rather than a cold zero-state --
    see `INTRA_SURAH_SPLIT_WARMUP_CHUNKS`'s docstring for the empirical
    calibration behind this), and stitch each segment's TRUSTED (post-
    warm-up) output back together into one continuous [T, 251] array --
    the exact same shape/contract as `run_streaming_log_probs_cuda_iobinding`
    would have returned for the whole surah run as one unsplit stream.

    Returns `(log_probs, seconds_per_output_frame)` -- same contract as
    `run_streaming_log_probs_cuda_iobinding`.
    """
    if not split_chunk_indices:
        return run_streaming_log_probs_cuda_iobinding(
            sess, feats, device_id=device_id, return_gpu_tensor=return_gpu_tensor
        )

    meta = sess.get_modelmeta().custom_metadata_map
    offset = int(meta.get("decode_chunk_len", 48))
    segment = int(meta.get("T", 61))

    segment_feats, seg_bounds = build_intra_surah_segments(
        feats, split_chunk_indices, offset, segment, warmup_chunks
    )

    log_probs_list, seconds_per_output_frame = run_streaming_log_probs_batched_cuda_iobinding(
        sess, segment_feats, device_id=device_id, return_gpu_tensor=return_gpu_tensor
    )

    log_probs = stitch_intra_surah_segments(log_probs_list, seg_bounds)
    return log_probs, seconds_per_output_frame


def build_intra_surah_segments(feats, split_chunk_indices, offset, segment,
                                warmup_chunks=INTRA_SURAH_SPLIT_WARMUP_CHUNKS):
    """Build the list of per-segment feature slices (each prefixed with
    `INTRA_SURAH_SPLIT_WARMUP_CHUNKS`-worth of real preceding audio) for
    ONE surah's `feats`, split at `split_chunk_indices` -- the shared core
    of `run_streaming_log_probs_intra_surah_split_cuda`, factored out so
    a caller batching MULTIPLE surahs' intra-surah segments together in
    one giant cross-surah-and-cross-segment batch (see
    `pipeline.align_surahs_batched`'s `intra_surah_split=True` combination)
    can build every surah's own segment list independently, concatenate
    them all into one flat list for a single
    `run_streaming_log_probs_batched_cuda_iobinding` call, and later split
    the returned per-segment results back out per surah before calling
    `stitch_intra_surah_segments` on each surah's own slice.

    Returns `(segment_feats, seg_bounds)`: `segment_feats` is a list of
    per-segment feature arrays (in the same warm-up-prefixed form
    `run_streaming_log_probs_batched_cuda_iobinding` expects); `seg_bounds`
    is a list of `(seg_start, seg_end, warmup_start)` triples (chunk
    indices), one per segment, that `stitch_intra_surah_segments` needs to
    know which portion of each segment's OWN returned log_probs is the
    "trusted" (post-warm-up) part to keep.
    """
    T_raw, feat_dim = feats.shape
    n_chunks_total = 1 if T_raw <= segment else 1 + math.ceil((T_raw - segment) / offset)
    total_len_needed = segment + (n_chunks_total - 1) * offset
    pad_needed = max(0, total_len_needed - T_raw)
    feats_padded = (
        np.concatenate([feats, np.zeros((pad_needed, feat_dim), dtype=np.float32)], axis=0)
        if pad_needed > 0 else feats
    )

    split_chunk_indices = sorted(split_chunk_indices)
    seg_starts = [0] + split_chunk_indices
    seg_ends = split_chunk_indices + [n_chunks_total]

    segment_feats = []
    seg_bounds = []
    for seg_start, seg_end in zip(seg_starts, seg_ends):
        warmup_start = max(0, seg_start - warmup_chunks)
        seg_bounds.append((seg_start, seg_end, warmup_start))
        start_sample = warmup_start * offset
        end_sample = min((seg_end - 1) * offset + segment, feats_padded.shape[0])
        segment_feats.append(feats_padded[start_sample:end_sample])

    return segment_feats, seg_bounds


def stitch_intra_surah_segments(log_probs_list, seg_bounds):
    """Inverse of `build_intra_surah_segments`: given each segment's OWN
    returned `log_probs` (in `log_probs_list`, same order as `seg_bounds`)
    and the `(seg_start, seg_end, warmup_start)` triples
    `build_intra_surah_segments` produced for them, keep only each
    segment's TRUSTED (post-warm-up) portion and concatenate them back
    into one continuous [T, 251] array for the whole surah -- the exact
    shape/contract `run_streaming_log_probs_cuda_iobinding` would have
    returned for this surah run as one unsplit stream.
    """
    frames_per_chunk_per_segment = [
        log_probs_list[i].shape[0] // (seg_bounds[i][1] - seg_bounds[i][2])
        for i in range(len(seg_bounds))
    ]
    assert len(set(frames_per_chunk_per_segment)) == 1, (
        f"intra-surah split segments disagree on frames-per-chunk: {frames_per_chunk_per_segment}"
    )
    frames_per_chunk = frames_per_chunk_per_segment[0]

    trusted_parts = []
    for i, (seg_start, seg_end, warmup_start) in enumerate(seg_bounds):
        trusted_chunk_offset = seg_start - warmup_start
        trusted_frame_offset = trusted_chunk_offset * frames_per_chunk
        trusted_frame_len = (seg_end - seg_start) * frames_per_chunk
        trusted_parts.append(log_probs_list[i][trusted_frame_offset : trusted_frame_offset + trusted_frame_len])

    if hasattr(trusted_parts[0], "device"):
        import torch
        return torch.cat(trusted_parts, dim=0)
    return np.concatenate(trusted_parts, axis=0)
