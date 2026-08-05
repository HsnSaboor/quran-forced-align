"""Determinism check: running the full pipeline twice on the same
(surah, audio) input must produce byte-identical output. Every source of
nondeterminism found during implementation was pinned (single-threaded
onnxruntime, dither=0 fbank, no unordered-dict/set dependence, float64
single-threaded Viterbi) -- see pipeline.align_surah's docstring.
"""
import os

from quran_forced_align.pipeline import align_surah

from .conftest import FIXTURES_DIR, MODEL_PATH, TOKENS_PATH


def _render_srt_text(cue_tuples):
    lines = []
    for idx, (word, start, end, sura, aya, is_repeat) in enumerate(cue_tuples, 1):
        from quran_forced_align.srt import fmt_srt_time
        tag = " [repeat]" if is_repeat else ""
        lines.append(f"{idx}\n{fmt_srt_time(start)} --> {fmt_srt_time(end)}\n{word}{tag}\n")
    return "\n".join(lines)


def test_determinism_surah1_al_fatiha():
    audio_path = os.path.join(FIXTURES_DIR, "001001_full.mp3")

    run1 = align_surah(1, audio_path, model_path=MODEL_PATH, tokens_path=TOKENS_PATH, verbose=False)
    run2 = align_surah(1, audio_path, model_path=MODEL_PATH, tokens_path=TOKENS_PATH, verbose=False)

    assert run1 == run2, "two runs of align_surah on the same input produced different cue tuples"

    text1 = _render_srt_text(run1)
    text2 = _render_srt_text(run2)
    assert text1 == text2, "rendered SRT text differs between two runs on the same input"

    assert len(run1) > 0, "expected at least one word cue for Al-Fatiha"
