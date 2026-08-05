"""Single source-of-truth pipeline: surah number + audio path -> word-level
cue tuples. Both `cli.py` (single-surah) and `batch_cli.py` (multi-surah,
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
from .audio import load_audio_as_wav16k
from .constants import DEFAULT_MAX_REPEAT_WINDOW_WORDS, MIN_WORD_DUR, SAMPLE_RATE
from .features import compute_fbank_features
from .onnx_model import make_onnx_session, run_streaming_log_probs
from .reference import build_combined_reference
from .repeats import detect_and_fix_repeats, extract_word_frame_spans
from .srt import cues_to_tuples
from .tokenizer import load_tokens
from .viterbi import ctc_forced_align, frame_spans_from_path


def align_surah(surah: int, audio_path: str, *, model_path: str, tokens_path: str,
                 anomaly_low_ratio: float = 0.15, anomaly_high_ratio: float = 3.0,
                 ayah_final_high_ratio_mult: float = 1.5, repeat_confidence_margin: float = 1.0,
                 max_repeat_window_words: int | None = DEFAULT_MAX_REPEAT_WINDOW_WORDS,
                 tail_silence_sec: float = 0.3, verbose: bool = True) -> list[tuple]:
    """Run the full forced-alignment + repeat-detection pipeline for one
    surah's audio and return its word-level cue tuples (word, start, end,
    sura, aya, is_repeat), matching what `srt.cues_to_tuples` produces.

    Keyword-arg defaults match the current CLI's argparse defaults exactly
    (see cli.py's --anomaly-low-ratio, --anomaly-high-ratio,
    --ayah-final-high-ratio-mult, --repeat-confidence-margin,
    --max-repeat-window-words, --tail-silence-sec).
    """
    def log(msg):
        if verbose:
            print(msg)

    log(f"[1/5] Building whole-surah word<->phoneme reference for surah {surah}...")
    tok2id, id2tok, blank_id, max_token_len = load_tokens(tokens_path)
    combined_token_ids, word_slots = build_combined_reference(surah, tok2id, max_token_len)
    log(f"      {len(word_slots)} words total, {len(combined_token_ids)} reference tokens")

    log(f"[2/5] Loading + extracting deterministic fbank features: {audio_path}")
    samples = load_audio_as_wav16k(audio_path)
    log(f"      {len(samples) / SAMPLE_RATE:.1f}s of audio")
    feats = compute_fbank_features(samples, tail_silence_sec=tail_silence_sec)

    log("[3/5] Running raw-ONNX streaming Zipformer2-CTC (cache-threaded chunks)...")
    sess = make_onnx_session(model_path)
    log_probs, seconds_per_frame = run_streaming_log_probs(sess, feats)
    log(f"      log_probs shape {log_probs.shape}, {seconds_per_frame * 1000:.1f}ms/output-frame")

    log("[4/5] CTC forced-alignment Viterbi over the WHOLE surah at once...")
    ext, path = ctc_forced_align(log_probs, combined_token_ids, blank_id)
    if ext is None:
        raise RuntimeError(
            "forced alignment failed: audio too short for this surah's reference "
            "(not enough frames to fit the blank-interleaved trellis)"
        )
    first_seen, last_seen = frame_spans_from_path(path, len(ext))
    cues = extract_word_frame_spans(word_slots, first_seen, last_seen)
    log(f"      {len(cues)}/{len(word_slots)} words got timing from the main pass")

    log("[5/5] Detecting + locally re-aligning repeats...")
    min_word_dur_frames = MIN_WORD_DUR / seconds_per_frame
    cues = detect_and_fix_repeats(
        cues, log_probs, combined_token_ids, blank_id, ext, path,
        anomaly_low_ratio, anomaly_high_ratio, min_word_dur_frames,
        ayah_final_high_ratio_mult=ayah_final_high_ratio_mult,
        confidence_margin=repeat_confidence_margin,
        max_repeat_window_words=max_repeat_window_words,
    )
    return cues_to_tuples(cues, seconds_per_frame)
