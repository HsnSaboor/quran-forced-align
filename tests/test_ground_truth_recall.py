"""Ground-truth precision/recall harness for repeats.detect_and_fix_repeats,
using the synthetic-but-real-audio test clips (test_A_full_natural.wav /
test_B_repeat_pause.wav / test_C_repeat_nopause.wav).

Ported from the original repo's ground_truth_test.py (a print-and-eyeball
script) into real pytest assertions, adapted to import from this package's
module structure instead of the flat monolithic script.

Does NOT go through the surah-driven pipeline.align_surah (reference.
build_surah_reference assumes a full-surah recitation). Instead this builds
the "word_slots" + "combined_token_ids" reference structure by hand, in
EXACTLY the shape reference.build_combined_reference() itself produces: a
flat list of token ids, and a parallel list of word-slot dicts each
carrying {"word","sura","aya","is_ayah_final","token_positions"} where
token_positions are indices into the flat token-id list.

Ground truth (ayah 2:255, words 0-6 and the repeated words 3-6 -- these
phoneme strings are lifted directly from the source repo's
model_v1/ordered_quran_phonemes.json 2:255 entry, hardcoded here as plain
data since that JSON file itself is not otherwise needed by this package):
  - test_A: WORDS[0:7] recited once, no repeat -> reference = WORDS[0:7]
  - test_B/test_C: WORDS[0:7] then a genuine repeat of WORDS[3:7]
    -> "doubled" reference = WORDS[0:7] + WORDS[3:7] (11 words)
  - test_B has a natural pause between the two utterances; test_C has none.
"""
import os
import wave

import numpy as np
import pytest

from quran_forced_align.constants import MIN_WORD_DUR
from quran_forced_align.engines.cpu import CPUEngine
from quran_forced_align.features import compute_fbank_features
from quran_forced_align.repeats import detect_and_fix_repeats, extract_word_frame_spans
from quran_forced_align.srt import cues_to_tuples
from quran_forced_align.tokenizer import load_tokens, tokenize_with_char_starts
from quran_forced_align.trellis import frame_spans_from_path

from .conftest import FIXTURES_DIR, MODEL_PATH, TOKENS_PATH

SAMPLE_RATE = 16000

# WORDS[0:7] and WORDS[3:7] of ayah 2:255 (Ayat al-Kursi), per-word phoneme
# strings as produced by quran_transcript's phonemizer -- see module
# docstring for provenance.
WORDS_0_7 = ["ءَللَااهُ", "لَاااا", "ءِلَااهَ", "ءِللَاا", "هُوَ", "لحَييُ", "لقَييُۥۥمُ"]
WORDS_3_7 = ["ءِللَاا", "هُوَ", "لحَييُ", "لقَييُۥۥمُ"]

# Timing tolerance for comparing the recall run's recovered repeat timing
# against the oracle run (fed the correct doubled reference directly).
# Both runs align the same audio deterministically but via different frame
# windows (oracle: whole-clip main-pass Viterbi; recall: localized
# doubled-reference re-alignment), so exact bit-identical frame indices
# aren't guaranteed -- a few-output-frame tolerance (~0.04s/frame) is the
# right bar, not exact equality.
TIMING_TOLERANCE_SEC = 0.15


def _load_wav16k_mono(path):
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE, wf.getframerate()
        assert wf.getnchannels() == 1, wf.getnchannels()
        assert wf.getsampwidth() == 2, wf.getsampwidth()
        n = wf.getnframes()
        data = wf.readframes(n)
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _build_manual_reference(ref_words, tok2id, max_token_len, sura=2, aya=255):
    """Build (combined_token_ids, word_slots) in the exact shape
    reference.build_combined_reference() produces, for an arbitrary list of
    already-known per-word phoneme strings (no ayah-boundary/istiaatha
    logic needed -- these are isolated test clips, not full surahs).

    `token_char_idx`/`letters` are populated with degenerate placeholder
    values (each token attributed to its own 1:1 "char", no real tajweed
    data) rather than left out: this test builds its word_slots by hand
    from raw phoneme strings with no underlying Uthmani text to derive
    real per-character tajweed/silent-letter data from, but
    extract_word_frame_spans (the real code under test here) requires both
    keys to be present on every slot -- see reference.build_combined_reference
    for what a real caller populates them with.
    """
    combined_token_ids = []
    word_slots = []
    for word_phonemes in ref_words:
        ids, _char_starts = tokenize_with_char_starts(word_phonemes, tok2id, max_token_len)
        start = len(combined_token_ids)
        positions = list(range(start, start + len(ids)))
        word_slots.append({
            "word": word_phonemes,
            "sura": sura,
            "aya": aya,
            "is_ayah_final": False,  # isolated clip, not a full ayah -- no waqf-lengthening exception applies
            "token_positions": positions,
            "token_char_idx": list(range(len(ids))),
            "letters": [{"char": tid, "deleted": False, "tajweed_rules": [], "boundary_tajweed_rules": []}
                        for tid in range(len(ids))],
        })
        combined_token_ids.extend(ids)
    return combined_token_ids, word_slots


def _run_one(audio_path, ref_words, tok2id, max_token_len,
             anomaly_low_ratio=0.15, anomaly_high_ratio=3.0,
             ayah_final_high_ratio_mult=1.5, confidence_margin=1.0):
    combined_token_ids, word_slots = _build_manual_reference(ref_words, tok2id, max_token_len)

    samples = _load_wav16k_mono(audio_path)
    feats = compute_fbank_features(samples, tail_silence_sec=0.3)

    engine = CPUEngine(MODEL_PATH)
    log_probs, seconds_per_frame = engine.run_inference(feats)

    ext, path, _margins = engine.forced_align(log_probs, combined_token_ids, tok2id["<blank>"])
    assert ext is not None, "forced alignment failed (audio too short for reference)"

    first_seen, last_seen = frame_spans_from_path(path, len(ext))
    cues = extract_word_frame_spans(word_slots, first_seen, last_seen)

    min_word_dur_frames = MIN_WORD_DUR / seconds_per_frame
    cues2 = detect_and_fix_repeats(
        engine, cues, log_probs, combined_token_ids, tok2id["<blank>"], ext, path,
        anomaly_low_ratio, anomaly_high_ratio, min_word_dur_frames,
        ayah_final_high_ratio_mult=ayah_final_high_ratio_mult,
        confidence_margin=confidence_margin,
    )
    return cues_to_tuples(cues2, seconds_per_frame)


@pytest.fixture(scope="module")
def tokens():
    return load_tokens(TOKENS_PATH)


def test_A_no_repeat_zero_flags(tokens):
    tok2id, id2tok, blank_id, max_token_len = tokens
    audio_path = os.path.join(FIXTURES_DIR, "test_A_full_natural.wav")
    cues = _run_one(audio_path, WORDS_0_7, tok2id, max_token_len)
    n_repeat_flags = sum(1 for c in cues if c[5])
    assert n_repeat_flags == 0, f"expected 0 repeat flags for non-repeating audio, got {n_repeat_flags}"
    assert len(cues) == 7, f"expected 7 word cues (no repeat), got {len(cues)}"


def _oracle_second_copy_timing(audio_path, tokens):
    """Feed the CORRECT doubled reference (words 0-6 + 3-6) directly, so
    the main-pass Viterbi itself resolves the second occurrence's timing --
    no repeat-detection guessing involved. This is the ground-truth oracle
    the recall run's recovered timing is compared against."""
    tok2id, id2tok, blank_id, max_token_len = tokens
    doubled_words = WORDS_0_7 + WORDS_3_7
    cues = _run_one(audio_path, doubled_words, tok2id, max_token_len)
    # last 4 cues (by construction, word slots 7..10) are the second copy of WORDS_3_7
    assert len(cues) == 11, f"expected 11 word cues from the doubled oracle reference, got {len(cues)}"
    second_copy = cues[-4:] if all(not c[5] for c in cues) else [c for c in cues if c[5]]
    return sorted(second_copy, key=lambda c: c[1])


def _recall_run_repeat_timing(audio_path, tokens):
    """Feed ONLY the correct (non-doubled) 7-word reference, pretending we
    don't know about the repeat -- exercises the actual anomaly-gate +
    K-window-search + acoustic-confidence-gate + gap-artifact-reject
    machinery in repeats.detect_and_fix_repeats."""
    tok2id, id2tok, blank_id, max_token_len = tokens
    cues = _run_one(audio_path, WORDS_0_7, tok2id, max_token_len)
    repeat_cues = sorted([c for c in cues if c[5]], key=lambda c: c[1])
    return cues, repeat_cues


def test_B_repeat_with_pause_recovered(tokens):
    audio_path = os.path.join(FIXTURES_DIR, "test_B_repeat_pause.wav")
    cues, repeat_cues = _recall_run_repeat_timing(audio_path, tokens)

    assert len(repeat_cues) == 4, (
        f"expected the full 4-word repeat (words 3-6) to be recovered with is_repeat=True, "
        f"got {len(repeat_cues)} repeat-flagged cues: {repeat_cues}"
    )
    assert [c[0] for c in repeat_cues] == WORDS_3_7, (
        f"recovered repeat words don't match expected WORDS[3:7]: {[c[0] for c in repeat_cues]}"
    )

    oracle = _oracle_second_copy_timing(audio_path, tokens)
    for recovered, expected in zip(repeat_cues, oracle):
        assert recovered[0] == expected[0], f"word mismatch: {recovered[0]!r} vs {expected[0]!r}"
        assert abs(recovered[1] - expected[1]) < TIMING_TOLERANCE_SEC, (
            f"{recovered[0]!r} start time {recovered[1]:.3f}s not within "
            f"{TIMING_TOLERANCE_SEC}s of oracle {expected[1]:.3f}s"
        )
        assert abs(recovered[2] - expected[2]) < TIMING_TOLERANCE_SEC, (
            f"{recovered[0]!r} end time {recovered[2]:.3f}s not within "
            f"{TIMING_TOLERANCE_SEC}s of oracle {expected[2]:.3f}s"
        )


def test_C_repeat_no_pause_recovered(tokens):
    audio_path = os.path.join(FIXTURES_DIR, "test_C_repeat_nopause.wav")
    cues, repeat_cues = _recall_run_repeat_timing(audio_path, tokens)

    assert len(repeat_cues) == 4, (
        f"expected the full 4-word repeat (words 3-6) to be recovered with is_repeat=True, "
        f"got {len(repeat_cues)} repeat-flagged cues: {repeat_cues}"
    )
    assert [c[0] for c in repeat_cues] == WORDS_3_7, (
        f"recovered repeat words don't match expected WORDS[3:7]: {[c[0] for c in repeat_cues]}"
    )

    oracle = _oracle_second_copy_timing(audio_path, tokens)
    for recovered, expected in zip(repeat_cues, oracle):
        assert recovered[0] == expected[0], f"word mismatch: {recovered[0]!r} vs {expected[0]!r}"
        assert abs(recovered[1] - expected[1]) < TIMING_TOLERANCE_SEC, (
            f"{recovered[0]!r} start time {recovered[1]:.3f}s not within "
            f"{TIMING_TOLERANCE_SEC}s of oracle {expected[1]:.3f}s"
        )
        assert abs(recovered[2] - expected[2]) < TIMING_TOLERANCE_SEC, (
            f"{recovered[0]!r} end time {recovered[2]:.3f}s not within "
            f"{TIMING_TOLERANCE_SEC}s of oracle {expected[2]:.3f}s"
        )


def _run_slots(audio_path, slots_ayah, full_comb_ids, tok2id,
               anomaly_low_ratio=0.15, anomaly_high_ratio=2.2,
               ayah_final_high_ratio_mult=1.3, confidence_margin=0.6):
    comb_ids_a = [full_comb_ids[p] for s in slots_ayah for p in s["token_positions"]]
    slots_adj = []
    cur_p = 0
    for s in slots_ayah:
        nt = len(s["token_positions"])
        s_c = dict(s)
        s_c["token_positions"] = list(range(cur_p, cur_p + nt))
        cur_p += nt
        slots_adj.append(s_c)

    samples = _load_wav16k_mono(audio_path)
    feats = compute_fbank_features(samples, tail_silence_sec=0.3)

    engine = CPUEngine(MODEL_PATH)
    log_probs, seconds_per_frame = engine.run_inference(feats)

    ext, path, _margins = engine.forced_align(log_probs, comb_ids_a, tok2id["<blank>"])
    assert ext is not None, "forced alignment failed"

    first_seen, last_seen = frame_spans_from_path(path, len(ext))
    cues = extract_word_frame_spans(slots_adj, first_seen, last_seen)

    min_word_dur_frames = MIN_WORD_DUR / seconds_per_frame
    cues2 = detect_and_fix_repeats(
        engine, cues, log_probs, comb_ids_a, tok2id["<blank>"], ext, path,
        anomaly_low_ratio=anomaly_low_ratio,
        anomaly_high_ratio=anomaly_high_ratio,
        min_word_dur_frames=min_word_dur_frames,
        ayah_final_high_ratio_mult=ayah_final_high_ratio_mult,
        confidence_margin=confidence_margin,
    )
    return cues_to_tuples(cues2, seconds_per_frame)


def test_surah2_ayah91_three_word_repeat(tokens):
    """Verify that Surah 2 Ayah 91 in Abdel-Mohsen Al-Obeikan recitation
    recovers the 3 repeated words: ['وَيَكْفُرُونَ', 'بِمَا', 'وَرَآءَهُۥ'].
    """
    tok2id, id2tok, blank_id, max_token_len = tokens
    from quran_forced_align.reference import build_combined_reference
    full_comb_ids, full_slots = build_combined_reference(2, tok2id, max_token_len, include_istiaatha=False)
    slots_91 = [s for s in full_slots if s["aya"] == 91]
    
    audio_path = os.path.join(FIXTURES_DIR, "test_surah2_ayah91.wav")
    cues = _run_slots(audio_path, slots_91, full_comb_ids, tok2id)
    repeats = [c for c in cues if c[5]]
    
    assert len(repeats) == 3, f"expected exactly 3 repeat words in Ayah 91, got {len(repeats)}: {[c[0] for c in repeats]}"
    expected_words = ["وَيَكْفُرُونَ", "بِمَا", "وَرَآءَهُۥ"]
    assert [c[0] for c in repeats] == expected_words, f"mismatch in Ayah 91 repeats: {[c[0] for c in repeats]} vs {expected_words}"


def test_surah2_ayah109_four_word_repeat(tokens):
    """Verify that Surah 2 Ayah 109 in Abdel-Mohsen Al-Obeikan recitation
    recovers EXACTLY the 4 repeated words: ['مِّنۢ', 'بَعْدِ', 'مَا', 'تَبَيَّنَ'].
    """
    tok2id, id2tok, blank_id, max_token_len = tokens
    from quran_forced_align.reference import build_combined_reference
    full_comb_ids, full_slots = build_combined_reference(2, tok2id, max_token_len, include_istiaatha=False)
    slots_109 = [s for s in full_slots if s["aya"] == 109]
    
    audio_path = os.path.join(FIXTURES_DIR, "test_surah2_ayah109.wav")
    cues = _run_slots(audio_path, slots_109, full_comb_ids, tok2id)
    repeats = [c for c in cues if c[5]]
    
    assert len(repeats) == 4, f"expected exactly 4 repeat words in Ayah 109, got {len(repeats)}: {[c[0] for c in repeats]}"
    expected_words = ["مِّنۢ", "بَعْدِ", "مَا", "تَبَيَّنَ"]
    assert [c[0] for c in repeats] == expected_words, f"mismatch in Ayah 109 repeats: {[c[0] for c in repeats]} vs {expected_words}"
