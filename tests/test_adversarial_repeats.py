"""Adversarial repeat suite: audio-edited real recitation (tests/adversarial/).

Each clip = carrier (test_B[0,8.0s]: words 0-6 + pause fragment) + an edited
copy of the genuine second-utterance phrase (test_B[8.40,12.61s]: words 3-6)
appended per tests/adversarial/manifest.json. Reference fed to the harness is
WORDS_0_7 (single-copy 7-word reference) -- identical harness settings to
tests/test_ground_truth_recall.py (anomaly 0.15/3.0, ayah-final mult 1.5,
confidence margin 1.0).

Categories from the manifest:
  must-flag    -> assert full 4-word repeat recovered (flag count + word identity)
  must-not     -> assert zero repeat flags
  characterize -> assert the LOCKED actual (flags + cue count); documents engine limits

No special setup: the 17 clips are checked in under tests/adversarial/, so plain
`pytest tests/` runs this suite. ffmpeg (6.1.1) is needed ONLY to rebuild clips
via tests/adversarial/make_adversarial.py; if a clip wav is absent the case
skips with a pointer to the builder instead of failing.

Calibration (/tmp/qlab/THRESHOLD_R_CHECK.md, 2026-10-07; gate QUO = RD 0.40 /
M -0.43 / K1M +0.05, constants.py:127-128):
  Proposed R (M -0.435, all else identical): 0 per-K flips over 87 rows /
  15 cells QUO -> R, only headroom shifts. Adversarial cross-check: 17/17 green
  under QUO, and all 17 verdicts hold under R by monotonicity -- (L1) every K
  passing under QUO still passes, so no must-flag gate pass can be lost; (L2) R
  can only ADD K-passes, so the only possible change is a 0->=1 flip on a
  currently-zero clip (paraphrase, reversed, partial2of4), requiring
  rd >= 0.40 AND margin in [-0.435, -0.43). None meets it: the zero clips are
  rd-bound or deep in the reject band (nearest measured FP approach anywhere
  is -0.4444, still rejects under R by 0.0094); the characterize locks (2.0x =
  2 flags, triple = 4/11 cues, ayahx = 2) are rd/pass/window-bound, not
  margin-bound. left128 cross-check under R: B +0.4462, C +0.4077, 67:28
  +0.3500, 68:26 reject +0.0650, 71:7 +0.2216 (release-C gate-h +0.0119, K3
  rd-boundary +0.0000). Formal gap: this suite logs flag counts/words/cues
  only, no per-K (rd, margin) rows -- dump those per clip on reruns so the
  cross-check can be recomputed exactly instead of via monotonicity.
"""
import json
import os
import time

import pytest

from quran_forced_align.constants import MIN_WORD_DUR
from quran_forced_align.engines.cpu import CPUEngine
from quran_forced_align.features import compute_fbank_features
from quran_forced_align.repeats import detect_and_fix_repeats, extract_word_frame_spans
from quran_forced_align.srt import cues_to_tuples
from quran_forced_align.tokenizer import load_tokens
from quran_forced_align.trellis import frame_spans_from_path

from .conftest import MODEL_PATH, TOKENS_PATH
from .test_ground_truth_recall import WORDS_0_7, _build_manual_reference, _load_wav16k_mono

ADV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adversarial")


def _load_manifest():
    with open(os.path.join(ADV_DIR, "manifest.json"), encoding="utf-8") as f:
        return json.load(f)["cases"]


@pytest.fixture(scope="module")
def tokens():
    return load_tokens(TOKENS_PATH)


@pytest.fixture(scope="module")
def engine():
    return CPUEngine(MODEL_PATH)


def _run_clip(audio_path, engine, tok2id, max_token_len):
    combined_token_ids, word_slots = _build_manual_reference(WORDS_0_7, tok2id, max_token_len)
    samples = _load_wav16k_mono(audio_path)
    feats = compute_fbank_features(samples, tail_silence_sec=0.3)
    log_probs, seconds_per_frame = engine.run_inference(feats)
    ext, path, _m = engine.forced_align(log_probs, combined_token_ids, tok2id["<blank>"])
    assert ext is not None, f"forced alignment failed for {audio_path}"
    first_seen, last_seen = frame_spans_from_path(path, len(ext))
    cues = extract_word_frame_spans(word_slots, first_seen, last_seen)
    cues2 = detect_and_fix_repeats(
        engine, cues, log_probs, combined_token_ids, tok2id["<blank>"], ext, path,
        0.15, 3.0, MIN_WORD_DUR / seconds_per_frame,
        ayah_final_high_ratio_mult=1.5, confidence_margin=1.0,
    )
    return cues_to_tuples(cues2, seconds_per_frame)


CASES = _load_manifest()
CASE_IDS = [c["file"] for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_adversarial_clip(case, tokens, engine):
    tok2id, _id2tok, _blank, max_token_len = tokens
    audio_path = os.path.join(ADV_DIR, case["file"])
    if not os.path.exists(audio_path):
        pytest.skip(f"missing clip {audio_path} -- rebuild with: ffmpeg + tests/adversarial/make_adversarial.py")
    cues = _run_clip(audio_path, engine, tok2id, max_token_len)
    reps = sorted([c for c in cues if c[5]], key=lambda c: c[1])
    rep_words = [c[0] for c in reps]
    print(f"\n[{case['file']}] cat={case['category']} flags={len(reps)} words={rep_words} ncues={len(cues)}")
    time.sleep(5)  # shared-box courtesy: cool down between CPU inference runs
    cat = case["category"]
    if cat == "must-flag":
        exp_words = case["expect"]["words"]
        assert len(reps) >= case["expect"]["min_flags"], (
            f"{case['file']}: expected >={case['expect']['min_flags']} repeat flags, got {len(reps)}: {reps}"
        )
        assert rep_words == exp_words, (
            f"{case['file']}: repeat words {rep_words} != expected {exp_words}"
        )
    elif cat == "must-not":
        assert len(reps) <= case["expect"]["max_flags"], (
            f"{case['file']}: expected no repeat flags, got {len(reps)}: {reps}"
        )
    elif cat == "characterize":
        assert len(reps) == case["actual"]["n_flags"], (
            f"{case['file']}: characterization drift -- locked {case['actual']['n_flags']} flags, "
            f"now {len(reps)}: {reps}. If the engine changed, re-lock deliberately, never silently."
        )
    else:
        raise AssertionError(f"unknown category {cat!r}")
