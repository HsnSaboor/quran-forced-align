"""Single source-of-truth pipeline: surah number + audio path -> word-level
cue records. Both `cli.py` (single-surah) and `batch_cli.py` (multi-surah,
multi-process) call `align_surah` -- no wiring logic is duplicated between
them.

WHY FORCED ALIGNMENT INSTEAD OF FREE-DECODE-THEN-MATCH
-------------------------------------------------------
The naive pipeline is: (1) free-decode the whole surah with greedy CTC (the
model picks whatever phonemes it thinks it heard, with no knowledge of what
SHOULD be there), then (2) banded-Levenshtein-align the decoded
phoneme+timestamp stream against the known per-ayah reference, walking a
cursor forward ayah by ayah. This is architecturally wrong for a
closed-vocabulary problem: the reference text is ALWAYS known in advance (we
know exactly which surah/ayah is being recited), so free decoding throws that
knowledge away and re-derives it via fuzzy matching. Any ASR substitution/
deletion/insertion mistake corrupts a timestamp, and -- because the cursor
only moves forward -- a bad match on ayah N can make the search window for
ayah N+1 start in the wrong place, and errors compound ayah-over-ayah.

CTC forced alignment fixes this by construction: given the FULL per-frame
CTC log-probability matrix (log P(symbol | frame) for all 251 symbols, not
just the greedily-decoded one) and the KNOWN reference phoneme sequence, we
run the standard CTC forced-alignment Viterbi (the same algorithm behind
torchaudio.functional.forced_align / "CTC segmentation") to find the single
globally-optimal monotonic mapping from reference-token-position to
audio-frame-position. The reference position and the audio position are
locked together by construction -- there is no cursor to drift, and no
"guess correctly first, then only be given credit if the guess also happens
to match" step. The result is a best-path frame-boundary assignment for
every reference phoneme, robust to imperfect acoustic model confidence
(forced alignment only needs the correct answer to have SOME probability
mass at roughly the right place, not to be the argmax -- a much easier bar
than open-vocabulary decoding).

WHY RAW ONNX + MANUAL CACHE THREADING INSTEAD OF sherpa_onnx's PYTHON API
--------------------------------------------------------------------------
This raw log-probability matrix is NOT obtainable via sherpa_onnx's Python
API. Exhaustively checked: OnlineRecognizer, the raw pybind11 module
sherpa_onnx.lib._sherpa_onnx, OnlineStream (only exposes accept_waveform,
input_finished, get_frames [fbank features, NOT model output], and option
getters/setters), and OnlineRecognizerResult/ys_probs (only exposes the
per-*decoded*-token log-prob of the greedily-chosen symbol, never the full
per-frame distribution over all 251 symbols). There is no hidden API, no
example script, and nothing in k2-fsa/sherpa-onnx that does CTC forced
alignment. The only way to get the full [T, 251] log-prob matrix is to load
the raw ONNX graph directly with onnxruntime and re-implement the streaming
Zipformer2-CTC chunk loop ourselves (see the model's own metadata:
model_type=zipformer2, decode_chunk_len=48, T=61), threading the 96
encoder-layer cache tensors + embed_states + processed_lens between calls
exactly as sherpa-onnx's own C++ streaming decoder does internally (confirmed
against k2-fsa/icefall's export-onnx-streaming.py convention).

WHOLE-SURAH-AT-ONCE VITERBI (not per-ayah)
-------------------------------------------
This implementation concatenates the ENTIRE surah's reference phonemes (all
ayahs, in order, plus the istiaatha preamble) into one token sequence and
runs ONE forced-alignment Viterbi pass against the FULL [T_total, 251]
log-prob matrix for the whole audio file. This requires no per-ayah
boundary-detection heuristics and no drift is possible, because Viterbi
finds the single globally-optimal alignment over the whole known reference
against the whole known audio in one shot.

REPEAT HANDLING
----------------
After the main whole-surah alignment, word durations are computed and
compared against the surah's median word duration. Words whose duration is
anomalously short (suggesting the reciter's repeat of that word got mostly
absorbed into a single sliver of audio) or anomalously long (suggesting a
repeat's audio got merged into one word's span) are flagged. For each
flagged word, forced alignment is re-run LOCALLY (only over the small frame
window between the flagged word's neighbours) against a DOUBLED reference
(that word's tokens, twice) -- if the doubled alignment finds two
well-separated, plausible-duration occurrences, the single cue is replaced
with two cues (one tagged is_repeat=True) and spliced back in. See
`repeats.detect_and_fix_repeats` for the full detail (including four
independently-verified correctness fixes on top of the original design).

OUTPUT RICHNESS: PHONEME/LETTER TIER + TAJWEED + CONFIDENCE (all free)
------------------------------------------------------------------------
Three additions layered on top of the base word-level alignment, all pure
post-processing over data the pipeline already computes -- none of them
re-run the ONNX model or the Viterbi DP, and none add a new O(T*M) pass:

  - Phoneme/letter tier: the Viterbi backtrace already produces a frame
    span for every individual reference TOKEN (see viterbi.frame_spans_from_path),
    not just every word -- `repeats.extract_word_frame_spans` used to
    collapse this to a word-level min/max and discard the per-token spans;
    it now keeps them (`token_frame_spans`), and `cells.build_letter_tier`
    groups them by which Uthmani letter each token belongs to (using
    `reference.py`'s already-computed phoneme-to-char mapping).
  - Tajweed rules + silent letters: `reference.build_text_reference`
    already calls `quran_transcript.quran_phonetizer`, which tags madd/
    qalqalah rules and silent (deleted) letters per Uthmani character --
    previously only `.pos` was read from its output; `.tajweed_rules`/
    `.deleted` are now threaded through into the letter tier too. A
    bounded per-ayah-boundary probe (`reference._boundary_bridge_rules`)
    additionally recovers wasl-only madd-rule changes at ayah boundaries
    that per-ayah phonetization alone would miss -- see that function's
    docstring for why a single whole-surah phonetizer call was tried and
    rejected (super-linear cost, unusable at Al-Baqarah's scale).
  - Confidence signals: `confidence.flag_low_confidence_words` surfaces two
    signals for every word -- the same average-log-probability quantity
    `repeats.py` already computed for repeat-anomaly screening (now
    computed for every word, not just anomalous ones), and a new
    "decision margin" (best-minus-second-best of the 3 candidate scores
    `viterbi._backtrack_step` already computes and discards per backtrace
    step). The margin is the closest zero-cost analogue an exact Viterbi
    DP has to a multi-beam-search-width disagreement signal -- see
    confidence.py's module docstring.

DETERMINISM
------------
Every source of nondeterminism found during implementation was pinned:
  - onnxruntime: intra_op_num_threads=1, inter_op_num_threads=1,
    execution_mode=ORT_SEQUENTIAL (rules out thread-race nondeterminism in
    parallelized reduction ops across identical runs).
  - kaldi_native_fbank: dither=0 explicitly set (the library's own default
    is 3e-05, i.e. nonzero -- this alone would make every run produce
    different features).
  - No use of Python set() or unordered dict iteration for anything
    order-sensitive; token maps are built from a linear file scan and
    dict insertion order is preserved (Python 3.7+, and not relied upon for
    anything beyond convenience anyway -- the tokenizer max-munge loop is
    driven by explicit descending-length range(), not dict order).
  - Viterbi DP is float64, single-threaded numpy; no reduction-order
    ambiguity since numpy's np.maximum over fixed-size arrays is elementwise
    and doesn't reassociate anything.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from .audio import load_audio_as_wav16k
from .confidence import flag_low_confidence_words
from .constants import (
    DEFAULT_ANOMALY_HIGH_RATIO,
    DEFAULT_ANOMALY_LOW_RATIO,
    DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
    DEFAULT_MAX_REPEAT_WINDOW_WORDS,
    DEFAULT_REPEAT_CONFIDENCE_MARGIN,
    DEFAULT_TAIL_SILENCE_SEC,
    FBANK_FRAME_SHIFT_SAMPLES,
    ISTIAATHA_DETECT_CONFIDENCE_THRESHOLD,
    ISTIAATHA_DETECT_MAX_FRAMES,
    ISTIAATHA_MIN_DETECT_FRAMES,
    MIN_WORD_DUR,
    SAMPLE_RATE,
)
from .engines import get_engine
from .features import compute_fbank_features, compute_fbank_features_gpu
from .reference import build_combined_reference
from .repeats import detect_and_fix_repeats, extract_word_frame_spans
from .silence import find_silence_midpoints
from .srt import build_rich_records
from .tokenizer import load_tokens
from .trellis import frame_spans_from_path


def _build_surah_inputs(surah, audio_path, tokens_path, tail_silence_sec, log, device="cpu", include_istiaatha=True):
    """Steps [1/6]-[2/6] of `align_surah`: build the word<->phoneme
    reference and extract fbank features for one surah -- factored out so
    `align_surahs_batched` can run this same per-surah preparation for
    every surah in a batch BEFORE the one batched inference call, without
    duplicating this logic.

    `tokens_path` may be either a path string to tokens.txt or an already-loaded
    `(tok2id, id2tok, blank_id, max_token_len)` tuple to avoid repeated disk reads.

    Returns `(tok2id, id2tok, blank_id, combined_token_ids, word_slots, feats, samples)`.
    `samples` (the raw 16kHz waveform) is returned alongside `feats` so
    `align_surah`'s intra-surah-split path can run silence detection on it
    without re-loading/re-decoding the audio file a second time.
    """
    log(f"[1/6] Building whole-surah word<->phoneme reference for surah {surah}...")
    t1_start = time.perf_counter()
    if isinstance(tokens_path, tuple) and len(tokens_path) == 4:
        tok2id, id2tok, blank_id, max_token_len = tokens_path
    else:
        tok2id, id2tok, blank_id, max_token_len = load_tokens(tokens_path)
    init_istiaatha = False if str(include_istiaatha).lower() in ("auto", "none") else bool(include_istiaatha)
    combined_token_ids, word_slots = build_combined_reference(surah, tok2id, max_token_len, include_istiaatha=init_istiaatha)
    t1_elapsed = time.perf_counter() - t1_start
    log(f"      {len(word_slots)} words total, {len(combined_token_ids)} reference tokens [{t1_elapsed:.3f}s]")

    log(f"[2/6] Loading + extracting deterministic fbank features: {audio_path}")
    t2_start = time.perf_counter()
    samples = load_audio_as_wav16k(audio_path)
    t2_audio = time.perf_counter() - t2_start
    audio_sec = len(samples) / SAMPLE_RATE
    log(f"      {audio_sec:.1f}s of audio (loaded in {t2_audio:.3f}s, {audio_sec/max(0.001,t2_audio):.0f}x realtime)")
    t2_fbank_start = time.perf_counter()
    if device == "cuda":
        try:
            feats = compute_fbank_features_gpu(samples, tail_silence_sec=tail_silence_sec, device="cuda")
        except Exception:
            feats = compute_fbank_features(samples, tail_silence_sec=tail_silence_sec)
    else:
        feats = compute_fbank_features(samples, tail_silence_sec=tail_silence_sec)
    t2_fbank = time.perf_counter() - t2_fbank_start
    t2_total = time.perf_counter() - t2_start
    log(f"      {feats.shape[0]} fbank frames (fbank: {t2_fbank:.3f}s, total stage 2: {t2_total:.3f}s)")
    return tok2id, id2tok, blank_id, combined_token_ids, word_slots, feats, samples


def detect_leading_istiaatha(log_probs, id2tok, blank_id=0, max_frames=ISTIAATHA_DETECT_MAX_FRAMES) -> bool:
    """Inspect the first ~8-10 seconds of acoustic log_probs to detect if
    the reciter recited Isti'adha (Audhubillah).
    Takes 0.01ms because log_probs is already computed.
    """
    import numpy as np
    T = min(len(log_probs), max_frames)
    if T < ISTIAATHA_MIN_DETECT_FRAMES:
        return False

    if hasattr(log_probs, "argmax"):
        if hasattr(log_probs, "is_cuda") and log_probs.is_cuda:
            argmax_ids = log_probs[:T].argmax(dim=-1).cpu().numpy()
        else:
            argmax_ids = np.argmax(log_probs[:T], axis=-1)
    else:
        argmax_ids = np.argmax(log_probs[:T], axis=-1)

    tokens = []
    prev = None
    for tid in argmax_ids:
        tid = int(tid)
        if tid != blank_id and tid != prev:
            tok = id2tok.get(tid, "")
            if tok:
                tokens.append(tok)
        prev = tid
    decoded = "".join(tokens)

    # Phonetic signatures of Isti'adha ('أعوذ', 'بالله', 'من', 'الشيطان', 'الرجيم')
    signatures = ("ءَعُ", "عُۥۥذُ", "عُذُ", "ششَي", "طَاانِ", "طَانِ", "جِۦۦم", "جِيم")
    return any(sig in decoded for sig in signatures)


def _align_from_log_probs(engine, log_probs, seconds_per_frame, combined_token_ids, blank_id,
                           word_slots, id2tok, anomaly_low_ratio, anomaly_high_ratio,
                           ayah_final_high_ratio_mult, repeat_confidence_margin,
                           max_repeat_window_words, log, silence_feature_frames=None,
                           strip_istiaatha=False):
    """Steps [4/6]-[6/6] of `align_surah`: forced-alignment, repeat
    detection, and confidence scoring, given an ALREADY-COMPUTED
    `log_probs` matrix.
    """
    import numpy as np
    t4_start = time.perf_counter()
    log("[4/6] CTC forced-alignment over the surah reference...")
    
    unique_ayahs = []
    for s in word_slots:
        if s["aya"] not in unique_ayahs:
            unique_ayahs.append(s["aya"])
            
    min_word_dur_frames = int(MIN_WORD_DUR / seconds_per_frame)
    T = log_probs.shape[0] if hasattr(log_probs, "shape") else len(log_probs)

    if len(unique_ayahs) > 1:
        # High precision per-ayah bounded alignment
        if hasattr(log_probs, "is_cuda") and log_probs.is_cuda:
            greedy_ids = log_probs.argmax(dim=-1).cpu().numpy()
        elif hasattr(log_probs, "argmax"):
            greedy_ids = log_probs.argmax(axis=-1)
            if hasattr(greedy_ids, "numpy"):
                greedy_ids = greedy_ids.numpy()
        else:
            greedy_ids = np.argmax(log_probs, axis=-1)

        is_non_blank = (greedy_ids != blank_id)
        prev_greedy = np.pad(greedy_ids[:-1], (1, 0), constant_values=blank_id)
        is_token_start = is_non_blank & ((greedy_ids != prev_greedy) | (prev_greedy == blank_id))
        token_frames = np.flatnonzero(is_token_start)
        total_c = len(token_frames)
        total_ref = len(combined_token_ids)
        
        all_cues = []
        t5_elapsed = 0.0
        t6_elapsed = 0.0

        for aya in unique_ayahs:
            slots_a = [s for s in word_slots if s["aya"] == aya]
            comb_ids_a = [combined_token_ids[p] for s in slots_a for p in s["token_positions"]]
            slots_adj = []
            cur_p = 0
            for s in slots_a:
                nt = len(s["token_positions"])
                s_c = dict(s)
                s_c["token_positions"] = list(range(cur_p, cur_p + nt))
                cur_p += nt
                slots_adj.append(s_c)

            ref_start = slots_a[0]["token_positions"][0]
            ref_end = slots_a[-1]["token_positions"][-1]

            c_s = int(ref_start * (total_c / max(1, total_ref)))
            c_e = int(ref_end * (total_c / max(1, total_ref)))
            f_s = max(0, int(token_frames[min(c_s, total_c - 1)]) - 125)
            f_e = min(T - 1, int(token_frames[min(c_e, total_c - 1)]) + 125)

            lp_a = log_probs[f_s:f_e + 1]
            ext_a, path_a, margins_a = engine.forced_align(lp_a, comb_ids_a, blank_id)
            if path_a is None:
                ext_a, path_a, margins_a = engine.forced_align(log_probs, comb_ids_a, blank_id)
                f_s = 0

            first_seen_a, last_seen_a = frame_spans_from_path(path_a, len(ext_a))
            cues_a = extract_word_frame_spans(slots_adj, first_seen_a, last_seen_a)
            t5_s = time.perf_counter()
            cues_a = detect_and_fix_repeats(
                engine, cues_a, lp_a, comb_ids_a, blank_id, ext_a, path_a,
                anomaly_low_ratio, anomaly_high_ratio, min_word_dur_frames,
                ayah_final_high_ratio_mult=ayah_final_high_ratio_mult,
                confidence_margin=repeat_confidence_margin,
                max_repeat_window_words=max_repeat_window_words,
            )
            t5_elapsed += (time.perf_counter() - t5_s)

            t6_s = time.perf_counter()
            cues_a = flag_low_confidence_words(cues_a, lp_a, ext_a, path_a, margins_a)
            t6_elapsed += (time.perf_counter() - t6_s)

            if f_s > 0 or ref_start > 0:
                for c in cues_a:
                    if f_s > 0:
                        c["start_frame"] += f_s
                        c["end_frame"] += f_s
                        if "token_spans" in c:
                            c["token_spans"] = [(st + f_s if st >= 0 else -1, en + f_s if en >= 0 else -1) for st, en in c["token_spans"]]
                    if ref_start > 0 and "token_positions" in c:
                        c["token_positions"] = [p + ref_start for p in c["token_positions"]]

            all_cues.extend(cues_a)

        t4_elapsed = time.perf_counter() - t4_start
        log(f"      {len(all_cues)}/{len(word_slots)} words aligned across {len(unique_ayahs)} ayahs [{t4_elapsed:.3f}s]")
        log(f"      Repeat detection complete [{t5_elapsed:.3f}s]")
        log(f"      Confidence scoring complete [{t6_elapsed:.3f}s]")
        return build_rich_records(all_cues, seconds_per_frame, combined_token_ids, id2tok, strip_istiaatha=strip_istiaatha)

    ext, path, margins = None, None, None
    if silence_feature_frames is not None and hasattr(engine, "forced_align_segmented"):
        ext, path, margins = engine.forced_align_segmented(
            log_probs, combined_token_ids, blank_id, silence_feature_frames, word_slots
        )
        if ext is not None:
            log("      (Segmented alignment successful)")

    if ext is None:
        ext, path, margins = engine.forced_align(log_probs, combined_token_ids, blank_id)
        if ext is None:
            raise RuntimeError(
                "forced alignment failed: audio too short for this surah's reference "
                "(not enough frames to fit the blank-interleaved trellis)"
            )
            
    first_seen, last_seen = frame_spans_from_path(path, len(ext))
    cues = extract_word_frame_spans(word_slots, first_seen, last_seen)
    t4_elapsed = time.perf_counter() - t4_start
    log(f"      {len(cues)}/{len(word_slots)} words got timing from the main pass [{t4_elapsed:.3f}s]")

    t5_start = time.perf_counter()
    log("[5/6] Detecting + locally re-aligning repeats...")
    cues = detect_and_fix_repeats(
        engine, cues, log_probs, combined_token_ids, blank_id, ext, path,
        anomaly_low_ratio, anomaly_high_ratio, min_word_dur_frames,
        ayah_final_high_ratio_mult=ayah_final_high_ratio_mult,
        confidence_margin=repeat_confidence_margin,
        max_repeat_window_words=max_repeat_window_words,
    )
    t5_elapsed = time.perf_counter() - t5_start
    log(f"      Repeat detection complete [{t5_elapsed:.3f}s]")

    t6_start = time.perf_counter()
    log("[6/6] Computing per-word alignment-confidence signals...")
    cues = flag_low_confidence_words(cues, log_probs, ext, path, margins)
    t6_elapsed = time.perf_counter() - t6_start
    log(f"      Confidence scoring complete [{t6_elapsed:.3f}s]")

    total = t4_elapsed + t5_elapsed + t6_elapsed
    log(f"      Stages 4-6 total: {total:.3f}s (align: {t4_elapsed:.3f}s, repeats: {t5_elapsed:.3f}s, confidence: {t6_elapsed:.3f}s)")

    return build_rich_records(cues, seconds_per_frame, combined_token_ids, id2tok, strip_istiaatha=strip_istiaatha)


def align_surah(surah: int, audio_path: str, *, model_path: str, tokens_path: str,
                 device: str = "cpu", intra_surah_split: bool | None = None,
                 include_istiaatha: bool | str = "auto",
                 anomaly_low_ratio: float = DEFAULT_ANOMALY_LOW_RATIO,
                 anomaly_high_ratio: float = DEFAULT_ANOMALY_HIGH_RATIO,
                 ayah_final_high_ratio_mult: float = DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
                 repeat_confidence_margin: float = DEFAULT_REPEAT_CONFIDENCE_MARGIN,
                 max_repeat_window_words: int | None = DEFAULT_MAX_REPEAT_WINDOW_WORDS,
                 tail_silence_sec: float = DEFAULT_TAIL_SILENCE_SEC, verbose: bool = True) -> list[dict]:
    """Run the full forced-alignment + repeat-detection pipeline for one
    surah's audio and return its word-level cue records, sorted by start
    time -- see `srt.build_rich_records` for the exact record shape (word/
    timing/repeat flag/confidence signals/letter-phoneme-tajweed tier).

    `device` selects the forced-alignment execution engine (`"cpu"`
    (default) or `"cuda"` -- see `engines/__init__.py`); both engines
    implement the identical `(ext, path, margins)` contract, so every step
    after `[3/6]` below is engine-agnostic.

    `intra_surah_split` (CUDA-only; ignored on `device="cpu"`): split THIS
    ONE surah's own acoustic-model inference into multiple warm-up-
    overlapped segments at real silence points, run together via the
    CUDA engine's batched-inference machinery, for real single-surah GPU
    speedup (verified empirically: ~2x on a real T4 GPU session for a
    ~6-minute surah split into 3 segments) -- see
    `engines.cuda.CUDAEngine.run_inference_intra_surah_split` and
    `onnx_model.run_streaming_log_probs_intra_surah_split_cuda`'s
    docstrings for the full rationale and the empirically-verified
    decode-level determinism characterization (word timings/repeat flags
    are unaffected; a tiny, decode-irrelevant floating-point difference in
    raw log_probs values is an unavoidable, verified-harmless artifact).
    Falls back to unsplit inference automatically if the audio has no
    usable silence gap (e.g. a short surah, or one recited with no
    internal pause at all).

    Keyword-arg defaults are the `constants.py` DEFAULT_* values cli.py's
    argparse defaults (--anomaly-low-ratio, --anomaly-high-ratio,
    --ayah-final-high-ratio-mult, --repeat-confidence-margin,
    --max-repeat-window-words, --tail-silence-sec) also read from, so the
    two can never silently drift apart.

    For batch-processing MANY surahs on a CUDA GPU, see
    `align_surahs_batched`, which shares steps [1/6]/[2/6]/[4/6]-[6/6]'s
    exact logic with this function (via `_build_surah_inputs`/
    `_align_from_log_probs`) but batches step [3/6]'s acoustic-model
    inference across surahs for real GPU throughput.
    """
    def log(msg):
        if verbose:
            print(msg, flush=True)

    t_e2e_start = time.perf_counter()

    if intra_surah_split is None:
        intra_surah_split = (device == "cuda")

    # --- Overlap model initialization with audio decode ---
    # Engine construction (ONNX session creation, GPU warm-up) takes several
    # seconds. Audio decode (Stage 2) takes ~30s on CPU. By running them
    # concurrently in a background thread, engine init is completely hidden
    # behind audio decode time — saving 2-4s of sequential wall-clock time.
    with ThreadPoolExecutor(max_workers=1) as pool:
        engine_future = pool.submit(lambda: get_engine(device)(model_path))

        tok2id, id2tok, blank_id, combined_token_ids, word_slots, feats, samples = (
            _build_surah_inputs(surah, audio_path, tokens_path, tail_silence_sec, log, device=device, include_istiaatha=include_istiaatha)
        )
        audio_sec = len(samples) / SAMPLE_RATE

        engine = engine_future.result()  # blocks only if model init took longer than audio decode

    t3_start = time.perf_counter()
    log(f"[3/6] Running streaming Zipformer2-CTC on the {device!r} engine (cache-threaded chunks)...")
    
    silence_feature_frames = None
    if intra_surah_split or hasattr(engine, "forced_align_segmented"):
        t_silence_start = time.perf_counter()
        silence_samples = find_silence_midpoints(samples, SAMPLE_RATE)
        silence_feature_frames = [pos // FBANK_FRAME_SHIFT_SAMPLES for pos in silence_samples]
        t_silence = time.perf_counter() - t_silence_start
        log(f"      found {len(silence_feature_frames)} candidate silence split point(s) [{t_silence:.3f}s]")

    if intra_surah_split and hasattr(engine, "run_inference_intra_surah_split"):
        log_probs, seconds_per_frame = engine.run_inference_intra_surah_split(feats, silence_feature_frames)
    else:
        log_probs, seconds_per_frame = engine.run_inference(feats)
    t3_elapsed = time.perf_counter() - t3_start
    lp_shape = log_probs.shape if hasattr(log_probs, 'shape') else f"tensor({log_probs.size()})"
    log(f"      log_probs shape {lp_shape}, {seconds_per_frame * 1000:.1f}ms/output-frame [{t3_elapsed:.3f}s, {audio_sec/max(0.001,t3_elapsed):.0f}x realtime]")

    # Resolve include_istiaatha == 'auto' via 0.01ms acoustic check on log_probs
    if str(include_istiaatha).lower() in ("auto", "none"):
        has_istiaatha = detect_leading_istiaatha(log_probs, id2tok)
        log(f"      [Auto-Isti'adha] Acoustic probe: {'Detected' if has_istiaatha else 'Not present'}")
        max_token_len = max(len(tok) for tok in tok2id)
        combined_token_ids, word_slots = build_combined_reference(surah, tok2id, max_token_len, include_istiaatha=has_istiaatha)
        strip_aya0 = True
    else:
        strip_aya0 = False

    records = _align_from_log_probs(
        engine, log_probs, seconds_per_frame, combined_token_ids, blank_id, word_slots, id2tok,
        anomaly_low_ratio, anomaly_high_ratio, ayah_final_high_ratio_mult,
        repeat_confidence_margin, max_repeat_window_words, log,
        silence_feature_frames=None,
        strip_istiaatha=strip_aya0,
    )
    t_e2e = time.perf_counter() - t_e2e_start
    log(f"      ════════════════════════════════════════════════════════════")
    log(f"      END-TO-END: {t_e2e:.3f}s for {audio_sec:.1f}s audio ({audio_sec/max(0.001,t_e2e):.1f}x realtime)")
    log(f"      ════════════════════════════════════════════════════════════")
    return records


def align_surahs_batched(surahs: list[int], audio_paths: list[str], *, model_path: str, tokens_path: str,
                          intra_surah_split: bool = False,
                          anomaly_low_ratio: float = DEFAULT_ANOMALY_LOW_RATIO,
                          anomaly_high_ratio: float = DEFAULT_ANOMALY_HIGH_RATIO,
                          ayah_final_high_ratio_mult: float = DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
                          repeat_confidence_margin: float = DEFAULT_REPEAT_CONFIDENCE_MARGIN,
                          max_repeat_window_words: int | None = DEFAULT_MAX_REPEAT_WINDOW_WORDS,
                          tail_silence_sec: float = DEFAULT_TAIL_SILENCE_SEC,
                          verbose: bool = True) -> list[list[dict]]:
    """CUDA-only batched sibling of `align_surah`: runs the acoustic-model
    inference step for ALL surahs in `surahs`/`audio_paths` through ONE
    streaming chunk loop (see `engines.cuda.CUDAEngine.run_inference_batched`
    and `onnx_model.run_streaming_log_probs_batched_cuda_iobinding` for the
    full rationale and ragged-length padding scheme), then runs each
    surah's own forced-alignment/repeat-detection/confidence steps

    `intra_surah_split=True` STACKS this cross-surah batching with the
    intra-surah silence-split optimization (see `align_surah`'s own
    `intra_surah_split` parameter): every surah's own audio is ALSO split
    into warm-up-overlapped segments at real silence points, and every
    surah's every segment is flattened into ONE giant cross-surah-and-
    cross-segment batch (see
    `engines.cuda.CUDAEngine.run_inference_batched_with_intra_surah_split`)
    -- the maximum-parallelism combination of both optimizations, for the
    100+ reciters x 114 surahs target workload where both many surahs AND
    each individual surah benefit from batching. Verified empirically
    (end-to-end word/timing/repeat output, not just log_probs closeness)
    to remain byte-identical-at-the-decode-level to the fully serial
    baseline, stacking the two independently-verified-harmless sources of
    floating-point drift described in each optimization's own docstring.
    (identical logic to `align_surah`'s steps [4/6]-[6/6], via
    `_align_from_log_probs`) sequentially against its own `log_probs`
    slice.

    Returns a list of per-surah word-cue-record lists, one per entry of
    `surahs`/`audio_paths`, in the same order.

    DETERMINISM: batched acoustic-model inference introduces a tiny
    (~1e-4-magnitude) numerical difference in raw log_probs values versus
    running the same audio alone or in a different batch -- verified
    empirically (on a real Colab T4 GPU session, across a full real surah
    and across genuinely different audio streams batched together) to
    NEVER change any argmax decision or `torchaudio.functional.forced_align`
    alignment path; every test found this function's actual OUTPUT
    (word timings, repeat flags, confidence signals) is unaffected by
    batching. Repeated calls with the SAME batch composition (same surahs,
    same order) reproduce identically. See
    `run_streaming_log_probs_batched_cuda_iobinding`'s docstring for the
    full characterization and for how to get the CPU/single-stream
    engines' stronger batch-composition-independent guarantee instead
    (call `align_surah` per surah, or use `device="cpu"`).
    """
    def log(msg):
        if verbose:
            print(msg)

    if len(surahs) != len(audio_paths):
        raise ValueError(
            f"align_surahs_batched: surahs (len {len(surahs)}) and audio_paths "
            f"(len {len(audio_paths)}) must be the same length"
        )

    engine = get_engine("cuda")(model_path)
    if not hasattr(engine, "run_inference_batched"):
        raise RuntimeError(
            "align_surahs_batched requires an engine with batched inference support "
            "(engines.cuda.CUDAEngine) -- got an engine with no run_inference_batched method"
        )

    per_surah_inputs = [
        _build_surah_inputs(surah, audio_path, tokens_path, tail_silence_sec, log, include_istiaatha=include_istiaatha)
        for surah, audio_path in zip(surahs, audio_paths)
    ]
    feats_list = [inputs[5] for inputs in per_surah_inputs]
    
    samples_list = [inputs[6] for inputs in per_surah_inputs]
    silence_frames_list = []
    if intra_surah_split or hasattr(engine, "forced_align_segmented"):
        silence_frames_list = [
            [pos // FBANK_FRAME_SHIFT_SAMPLES for pos in find_silence_midpoints(samples, SAMPLE_RATE)]
            for samples in samples_list
        ]

    if intra_surah_split and hasattr(engine, "run_inference_batched_with_intra_surah_split"):
        total_splits = sum(1 for frames in silence_frames_list if frames)
        log(f"[3/6 batched+split] Running streaming Zipformer2-CTC for {len(surahs)} surahs "
            f"({total_splits} with usable silence splits) in one combined batched pass...")
        log_probs_list, seconds_per_frame = engine.run_inference_batched_with_intra_surah_split(
            feats_list, silence_frames_list
        )
    else:
        log(f"[3/6 batched] Running streaming Zipformer2-CTC for {len(surahs)} surahs in one batched pass...")
        log_probs_list, seconds_per_frame = engine.run_inference_batched(feats_list)

    results = []
    for idx, (surah, (tok2id, id2tok, blank_id, combined_token_ids, word_slots, _feats, _samples), log_probs) in enumerate(zip(
        surahs, per_surah_inputs, log_probs_list
    )):
        log(f"-- surah {surah} --")
        # `log_probs` here is a VIEW into `run_inference_batched`'s shared
        # batched buffer (log_probs_batched[i, :n, :]) -- its own `.base`
        # points at THAT shared buffer, not at `log_probs` itself. Passing
        # it to `_align_from_log_probs`/`repeats/candidate.py` as-is would
        # make every later `log_probs[window_start:window_end+1]` slice's
        # `.base` ALSO point at the shared batched buffer (numpy collapses
        # view-of-a-view chains), never matching whatever
        # `engine._last_log_probs_cpu` is set to below -- silently
        # defeating `_as_device_tensor`'s GPU-residency fast path for
        # every repeat-detection candidate in the batched pipeline (a real
        # bug found in code review: the fast path was live and correct
        # for the single-surah/unbatched path, but always fell through to
        # a full host->device re-upload here instead). Copying once here
        # breaks the view chain: `log_probs` becomes its OWN base array
        # (`.base is None`), so every later slice of it correctly matches
        # the cache set immediately below, restoring the intended
        # optimization for this pipeline too.
        log_probs = log_probs.copy()
        # Re-establish this engine's single-stream GPU-residency cache
        # (see engines.cuda.CUDAEngine.run_inference/_as_device_tensor) for
        # THIS surah's own log_probs slice, so its own repeat-detection
        # candidates below still get the on-device-slice fast path instead
        # of re-uploading from host on every K-search call -- the batched
        # inference call above does not (and cannot generically) populate
        # this per-surah cache itself, since it returns N separate slices
        # of one shared batched buffer, not N independently-cached arrays.
        if hasattr(engine, "_last_log_probs_cpu"):
            engine._last_log_probs_cpu = log_probs
            engine._last_log_probs_gpu = engine._torch.as_tensor(
                log_probs, dtype=engine._torch.float32, device=engine._device
            )
            
        silence_feature_frames = silence_frames_list[idx] if silence_frames_list else None

        if str(include_istiaatha).lower() in ("auto", "none"):
            has_istiaatha = detect_leading_istiaatha(log_probs, id2tok)
            combined_token_ids, word_slots = build_combined_reference(surah, tok2id, max_token_len, include_istiaatha=has_istiaatha)
            strip_aya0 = True
        else:
            strip_aya0 = False

        records = _align_from_log_probs(
            engine, log_probs, seconds_per_frame, combined_token_ids, blank_id, word_slots, id2tok,
            anomaly_low_ratio, anomaly_high_ratio, ayah_final_high_ratio_mult,
            repeat_confidence_margin, max_repeat_window_words, log,
            silence_feature_frames=silence_feature_frames,
            strip_istiaatha=strip_aya0,
        )
        results.append(records)
    return results
