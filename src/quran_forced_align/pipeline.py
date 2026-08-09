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
from .audio import load_audio_as_wav16k
from .confidence import flag_low_confidence_words
from .constants import (
    DEFAULT_ANOMALY_HIGH_RATIO,
    DEFAULT_ANOMALY_LOW_RATIO,
    DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
    DEFAULT_MAX_REPEAT_WINDOW_WORDS,
    DEFAULT_REPEAT_CONFIDENCE_MARGIN,
    DEFAULT_TAIL_SILENCE_SEC,
    MIN_WORD_DUR,
    SAMPLE_RATE,
)
from .engines import get_engine
from .features import compute_fbank_features
from .reference import build_combined_reference
from .repeats import detect_and_fix_repeats, extract_word_frame_spans
from .srt import build_rich_records
from .tokenizer import load_tokens
from .trellis import frame_spans_from_path


def align_surah(surah: int, audio_path: str, *, model_path: str, tokens_path: str,
                 device: str = "cpu",
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

    Keyword-arg defaults are the `constants.py` DEFAULT_* values cli.py's
    argparse defaults (--anomaly-low-ratio, --anomaly-high-ratio,
    --ayah-final-high-ratio-mult, --repeat-confidence-margin,
    --max-repeat-window-words, --tail-silence-sec) also read from, so the
    two can never silently drift apart.
    """
    def log(msg):
        if verbose:
            print(msg)

    log(f"[1/6] Building whole-surah word<->phoneme reference for surah {surah}...")
    tok2id, id2tok, blank_id, max_token_len = load_tokens(tokens_path)
    combined_token_ids, word_slots = build_combined_reference(surah, tok2id, max_token_len)
    log(f"      {len(word_slots)} words total, {len(combined_token_ids)} reference tokens")

    log(f"[2/6] Loading + extracting deterministic fbank features: {audio_path}")
    samples = load_audio_as_wav16k(audio_path)
    log(f"      {len(samples) / SAMPLE_RATE:.1f}s of audio")
    feats = compute_fbank_features(samples, tail_silence_sec=tail_silence_sec)

    log(f"[3/6] Running streaming Zipformer2-CTC on the {device!r} engine (cache-threaded chunks)...")
    engine = get_engine(device)(model_path)
    log_probs, seconds_per_frame = engine.run_inference(feats)
    log(f"      log_probs shape {log_probs.shape}, {seconds_per_frame * 1000:.1f}ms/output-frame")

    log("[4/6] CTC forced-alignment over the WHOLE surah at once...")
    ext, path, margins = engine.forced_align(log_probs, combined_token_ids, blank_id)
    if ext is None:
        raise RuntimeError(
            "forced alignment failed: audio too short for this surah's reference "
            "(not enough frames to fit the blank-interleaved trellis)"
        )
    first_seen, last_seen = frame_spans_from_path(path, len(ext))
    cues = extract_word_frame_spans(word_slots, first_seen, last_seen)
    log(f"      {len(cues)}/{len(word_slots)} words got timing from the main pass")

    log("[5/6] Detecting + locally re-aligning repeats...")
    min_word_dur_frames = MIN_WORD_DUR / seconds_per_frame
    cues = detect_and_fix_repeats(
        engine, cues, log_probs, combined_token_ids, blank_id, ext, path,
        anomaly_low_ratio, anomaly_high_ratio, min_word_dur_frames,
        ayah_final_high_ratio_mult=ayah_final_high_ratio_mult,
        confidence_margin=repeat_confidence_margin,
        max_repeat_window_words=max_repeat_window_words,
    )

    log("[6/6] Computing per-word alignment-confidence signals...")
    cues = flag_low_confidence_words(cues, log_probs, ext, path, margins)

    return build_rich_records(cues, seconds_per_frame, combined_token_ids, id2tok)
