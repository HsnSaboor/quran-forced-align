"""Determinism check: running the full pipeline twice on the same
(surah, audio) input must produce byte-identical output. Every source of
nondeterminism found during implementation was pinned (single-threaded
onnxruntime, dither=0 fbank, no unordered-dict/set dependence, float64
single-threaded Viterbi) -- see pipeline.align_surah's docstring.
"""
import json
import os

from quran_forced_align.pipeline import align_surah
from quran_forced_align.srt import emit_srt

from .conftest import FIXTURES_DIR, MODEL_PATH, TOKENS_PATH


def test_determinism_surah1_al_fatiha(tmp_path):
    audio_path = os.path.join(FIXTURES_DIR, "001001_full.mp3")

    run1 = align_surah(1, audio_path, model_path=MODEL_PATH, tokens_path=TOKENS_PATH, verbose=False)
    run2 = align_surah(1, audio_path, model_path=MODEL_PATH, tokens_path=TOKENS_PATH, verbose=False)

    # Guard against a vacuously-"deterministic" false pass (e.g. a bug that
    # makes align_surah always return []) BEFORE the equality checks below,
    # which would otherwise pass trivially on two empty lists.
    assert len(run1) > 0, "expected at least one word cue for Al-Fatiha"

    # Primary check: direct Python equality on the full rich-record output
    # (nested letter/phoneme tiers, confidence floats, everything) -- this
    # is list-order-sensitive (unlike a sort_keys=True JSON dump, which
    # only normalizes dict KEY order, not list order) and catches float
    # inequality directly rather than through a string round-trip.
    assert run1 == run2, "two runs of align_surah on the same input produced different records"

    # Secondary check: JSON round-tripping itself must also be
    # deterministic (e.g. no accidental float-repr nondeterminism
    # introduced only at serialization time, which `run1 == run2` alone
    # wouldn't catch since it never serializes anything).
    text1 = json.dumps(run1, ensure_ascii=False)
    text2 = json.dumps(run2, ensure_ascii=False)
    assert text1 == text2, "two runs of align_surah produced identical records but different JSON serializations"

    srt_path1 = tmp_path / "run1.srt"
    srt_path2 = tmp_path / "run2.srt"
    emit_srt(run1, str(srt_path1))
    emit_srt(run2, str(srt_path2))
    assert srt_path1.read_text() == srt_path2.read_text(), "rendered SRT text differs between two runs on the same input"
