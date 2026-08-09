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


def run_streaming_log_probs_cuda_iobinding(sess, feats, device_id=0):
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
        sess, [feats], device_id=device_id
    )
    return log_probs_list[0], seconds_per_output_frame


def run_streaming_log_probs_batched_cuda_iobinding(sess, feats_list, device_id=0):
    """Batched-N variant of `run_streaming_log_probs_cuda_iobinding`: runs
    `len(feats_list)` independent audio streams through the SAME streaming
    Zipformer2-CTC graph, stacked along the model's own dynamic `N`
    (batch) axis, instead of one fully-serial `sess.run` chunk loop per
    stream.

    WHY THIS EXISTS: `N` is already a dynamic axis on every input/output of
    this model's ONNX graph (confirmed via `sess.get_inputs()`: `x` is
    `['N', 61, 80]`, every cache tensor carries an `'N'` axis too) -- this
    is the same graph-level capability sherpa-onnx's own C++ streaming
    decoder and icefall's `streaming_decode.py` use to batch multiple
    independent utterances through one encoder call
    (`decode_streams`/`--num-decode-streams`); nothing about the ONNX graph
    itself needs to change, only how many streams' worth of cache tensors
    get stacked into each call.

    RAGGED LENGTHS: real surahs have wildly different total chunk counts
    (`n_chunks` varies by up to ~30x across the 114-surah/100-reciter
    target workload this exists for). This function pads every stream's
    feature array to `max(n_chunks across the batch)` with trailing
    zero (silence) frames -- the SAME padding scheme `run_streaming_log_probs`
    already uses for a single stream's last chunk, just extended to
    however many WHOLE extra chunks a shorter stream needs to reach the
    batch's longest stream's chunk count -- rather than the more
    complex "dynamically swap in a new stream when an old one finishes"
    scheme icefall's own decoder uses for a long-lived server process.
    That scheme exists there to keep a permanently-running batch full
    across many INCOMING streams over time; this package's batch_cli.py
    instead has a FIXED, known-up-front list of surahs per invocation, so
    the extra-compute cost of padding to the batch max (wasted cycles on
    already-finished streams' padding chunks) is bounded and predictable,
    not worth the substantial extra bookkeeping complexity of dynamic
    stream swapping for this package's actual (bounded, offline) workload
    shape.

    DETERMINISM: verified empirically (on a real Colab T4 GPU session)
    that batched (N>1) inference introduces a tiny (~1e-5 to ~3e-4
    magnitude, growing with sequence length as expected from a recurrent
    cache) numerical difference in raw log_probs values versus running the
    SAME audio alone (N=1) -- this comes from CUDA kernels reducing
    across the batch dimension in a different order than the N=1 case, not
    from any cross-contamination between streams' actual data. Also
    verified: across a full real surah (8952 frames) and across two
    genuinely different audio streams batched together, this drift NEVER
    flipped a single argmax decision, and NEVER changed
    `torchaudio.functional.forced_align`'s resulting alignment path (the
    actual output this whole pipeline cares about). This means: for a
    FIXED batch composition (the same set of surahs run together, in the
    same batch positions), output is exactly as deterministic as ever
    (repeated runs of that exact batch reproduce identically -- verified).
    Running the SAME surah in a DIFFERENT batch (different companions,
    different batch size) can in principle produce a log_probs value that
    differs at the ~1e-4 level from running it alone or in a different
    batch -- but every empirical test performed found this never changes
    the DECODED/ALIGNED output, only the never-directly-consumed raw
    log-probability magnitude. Anyone needing the CPU/single-stream
    engine's stronger byte-for-byte-regardless-of-batching guarantee
    should use `--cuda-batch-size 1` (equivalent to the unbatched
    `run_streaming_log_probs_cuda_iobinding` path) or the CPU engine.

    Returns `(log_probs_list, seconds_per_output_frame)`: `log_probs_list[i]`
    is stream `i`'s own [T_i, 251] float32 array, already truncated to that
    stream's real (unpadded) chunk count -- callers never see the padding
    chunks used internally to square off the batch.
    """
    meta = sess.get_modelmeta().custom_metadata_map
    offset = int(meta.get("decode_chunk_len", 48))
    segment = int(meta.get("T", 61))

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

    padded_feats = []
    for feats, n_chunks in zip(feats_list, n_chunks_per_stream):
        total_len_needed = segment + (n_chunks_max - 1) * offset
        pad_needed = max(0, total_len_needed - feats.shape[0])
        if pad_needed > 0:
            feats = np.concatenate([feats, np.zeros((pad_needed, feat_dim), dtype=np.float32)], axis=0)
        padded_feats.append(feats)

    io_binding = sess.io_binding()
    for inp in inputs:
        if inp.name == "x":
            continue
        if inp.name == "processed_lens":
            arr = np.zeros((N,), dtype=np.int64)
        else:
            dims = [N if d == "N" else d for d in inp.shape]
            arr = np.zeros(tuple(dims), dtype=np.float32)
        io_binding.bind_ortvalue_input(inp.name, ort.OrtValue.ortvalue_from_numpy(arr, "cuda", device_id))
    io_binding.bind_output("log_probs", "cuda", device_id)
    for name in state_names:
        io_binding.bind_output("new_" + name, "cuda", device_id)

    log_probs_batched = None  # preallocated once the first chunk reveals frames_per_chunk
    ptr = 0
    for chunk_idx in range(n_chunks_max):
        chunk_np = np.stack(
            [feats[ptr:ptr + segment] for feats in padded_feats], axis=0
        ).astype(np.float32, copy=False)
        io_binding.bind_ortvalue_input("x", ort.OrtValue.ortvalue_from_numpy(chunk_np, "cuda", device_id))

        sess.run_with_iobinding(io_binding)
        outs = io_binding.get_outputs()

        chunk_log_probs = outs[log_probs_out_idx].numpy()  # [N, frames_per_chunk, 251]
        if log_probs_batched is None:
            frames_per_chunk = chunk_log_probs.shape[1]
            log_probs_batched = np.empty(
                (N, n_chunks_max * frames_per_chunk, chunk_log_probs.shape[2]), dtype=np.float32
            )
        assert chunk_log_probs.shape[1] == frames_per_chunk, (
            f"chunk {chunk_idx} emitted {chunk_log_probs.shape[1]} frames per stream, "
            f"expected fixed frames_per_chunk={frames_per_chunk} (from chunk 0)"
        )
        row_start = chunk_idx * frames_per_chunk
        log_probs_batched[:, row_start:row_start + frames_per_chunk, :] = chunk_log_probs

        for name, out_idx in zip(state_names, state_out_idx):
            io_binding.bind_ortvalue_input(name, outs[out_idx])
        io_binding.bind_output("log_probs", "cuda", device_id)
        for name in state_names:
            io_binding.bind_output("new_" + name, "cuda", device_id)
        ptr += offset

    subsample_factor = offset / frames_per_chunk  # empirically 48/12 = 4
    seconds_per_output_frame = FRAME_SHIFT_SEC * subsample_factor

    log_probs_list = [
        log_probs_batched[i, :n_chunks_per_stream[i] * frames_per_chunk, :]
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
INTRA_SURAH_SPLIT_WARMUP_CHUNKS = 100


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
                                                     device_id=0):
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

    WHY THIS IS SAFE (unlike naively resetting the cache mid-stream with
    NO warm-up, which measurably corrupts the model's output near the
    reset point -- verified empirically: 5 argmax flips in the first 50
    frames after a naive zero-warmup reset, log-prob differences up to
    ~14 nats): this model's streaming cache only ever needs a BOUNDED
    left-context window (`left_context_len` in the model's own ONNX
    metadata: 256/128/64/32/64/128 frames per encoder stage, not
    unbounded history) -- so feeding enough REAL preceding audio through
    a fresh zero-state before trusting a segment's output lets the model
    rebuild an operationally-equivalent cache from scratch, rather than
    genuinely needing every frame since the start of the recording.
    `INTRA_SURAH_SPLIT_WARMUP_CHUNKS` is calibrated with a large safety
    margin above the smallest warm-up window empirically found sufficient.

    Splitting at a genuine SILENCE point (see `silence.find_silence_midpoints`/
    `choose_intra_surah_split_points`) rather than an arbitrary chunk
    boundary is not required for correctness (arbitrary mid-word split
    points were ALSO verified to reach zero decode-level difference given
    the same warm-up window), but silence points reach that same
    zero-difference bar with a SMALLER warm-up window in practice
    (verified empirically: a real silence-point split needed only ~30
    warm-up chunks to reach zero argmax flips, vs ~40 for an arbitrary
    chunk boundary in the same recording) -- splitting at silence is the
    more efficient choice, not a correctness requirement.

    DETERMINISM: verified empirically (on a real Colab T4 GPU session,
    across 4 different real surahs and split counts K=2 through 5) that
    `torchaudio.functional.forced_align`'s resulting alignment path over
    this function's output is IDENTICAL, every single time, to the path
    over the unsplit continuous-stream `log_probs` for the same audio --
    14/14 test configurations passed with ZERO argmax-level or
    alignment-level differences (`INTRA_SURAH_SPLIT_WARMUP_CHUNKS`'s
    margin above the empirically-found failure threshold is why). The raw
    log_probs VALUES do carry a tiny (~1e-4 magnitude) floating-point
    difference from the continuous run -- an unavoidable floating-point
    non-associativity artifact of running a different-length recurrence
    through the same recurrent math (the identical phenomenon already
    documented for cross-surah batching in
    `run_streaming_log_probs_batched_cuda_iobinding`'s docstring, here
    triggered by recursion length instead of batch position) -- but this
    was verified, in every test performed, to never change any argmax
    decision or forced-alignment result, which is the only thing this
    pipeline's output actually depends on.

    Returns `(log_probs, seconds_per_output_frame)` -- same contract as
    `run_streaming_log_probs_cuda_iobinding`.
    """
    if not split_chunk_indices:
        return run_streaming_log_probs_cuda_iobinding(sess, feats, device_id=device_id)

    meta = sess.get_modelmeta().custom_metadata_map
    offset = int(meta.get("decode_chunk_len", 48))
    segment = int(meta.get("T", 61))

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

    # Build each segment's own feature slice: from `warmup_chunks` chunks
    # before its real start (clamped to 0 for the first segment, which
    # needs no warm-up at all -- it already starts at the true beginning
    # of the recording) through its real end. A segment spanning chunk
    # indices [warmup_start, seg_end) has LAST real chunk index
    # `seg_end - 1`, which needs `segment` feature frames starting at
    # `(seg_end - 1) * offset` -- NOT `seg_end * offset` (an off-by-one-
    # chunk bug found via live end-to-end testing: the wrong, one-chunk-
    # too-long formula silently included one extra chunk's worth of
    # frames per segment, which the "trusted" truncation below never
    # caught since it only knows about warm-up/real-start boundaries, not
    # real-END boundaries -- the extra chunk's frames leaked into the
    # stitched output as a spurious, constant 1-chunk time-shift from that
    # segment boundary onward. Caught by comparing intra-surah-split
    # output against the unsplit baseline end-to-end, not by the
    # log_probs-only unit-level checks alone).
    segment_feats = []
    warmup_starts = []
    for seg_start, seg_end in zip(seg_starts, seg_ends):
        warmup_start = max(0, seg_start - warmup_chunks)
        warmup_starts.append(warmup_start)
        start_sample = warmup_start * offset
        end_sample = min((seg_end - 1) * offset + segment, feats_padded.shape[0])
        segment_feats.append(feats_padded[start_sample:end_sample])

    log_probs_list, seconds_per_output_frame = run_streaming_log_probs_batched_cuda_iobinding(
        sess, segment_feats, device_id=device_id
    )

    frames_per_chunk_per_segment = [
        log_probs_list[i].shape[0] // (seg_ends[i] - warmup_starts[i])
        for i in range(len(seg_starts))
    ]
    # Every segment's frames-per-chunk must agree (same model, same fixed
    # per-chunk output size regardless of which chunk-loop call produced
    # it) -- fail loudly rather than silently misassign trusted-frame
    # boundaries if this were ever violated.
    assert len(set(frames_per_chunk_per_segment)) == 1, (
        f"intra-surah split segments disagree on frames-per-chunk: {frames_per_chunk_per_segment}"
    )
    frames_per_chunk = frames_per_chunk_per_segment[0]

    trusted_parts = []
    for i, (seg_start, seg_end, warmup_start) in enumerate(zip(seg_starts, seg_ends, warmup_starts)):
        trusted_chunk_offset = seg_start - warmup_start
        trusted_frame_offset = trusted_chunk_offset * frames_per_chunk
        trusted_parts.append(log_probs_list[i][trusted_frame_offset:])

    log_probs = np.concatenate(trusted_parts, axis=0)
    return log_probs, seconds_per_output_frame
