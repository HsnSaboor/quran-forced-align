"""Unit tests for silence.find_silence_midpoints (energy-based silence-
boundary detection) and onnx_model.choose_intra_surah_split_points (split-
point selection with warm-up-window spacing constraints) -- pure numpy,
no GPU/model file needed, so these run unconditionally on every install.
"""
import numpy as np

from quran_forced_align.onnx_model import choose_intra_surah_split_points
from quran_forced_align.silence import find_silence_midpoints


def _tone(duration_sec, amplitude, sample_rate=16000, freq=200):
    n = int(duration_sec * sample_rate)
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_sec, sample_rate=16000):
    return np.zeros(int(duration_sec * sample_rate), dtype=np.float32)


def test_find_silence_midpoints_detects_a_real_gap():
    # Several loud/silence alternations so the silent frames make up a large
    # enough share of the whole recording's energy distribution to fall
    # below the bottom-5th-percentile threshold (a single short silence
    # gap in an otherwise long recording would not, by construction of a
    # PERCENTILE-based threshold -- this mirrors the real-world case of a
    # multi-ayah surah with several genuine pauses, not a single isolated one).
    samples = np.concatenate([
        _tone(1.0, 0.5), _silence(1.0),
        _tone(1.0, 0.5), _silence(1.0),
        _tone(1.0, 0.5), _silence(1.0),
        _tone(1.0, 0.5),
    ])
    midpoints = find_silence_midpoints(samples, sample_rate=16000)
    assert len(midpoints) >= 1
    # every found midpoint should land inside one of the three silent regions:
    # [1.0,2.0], [3.0,4.0], [5.0,6.0]
    silent_regions = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    for m in midpoints:
        m_sec = m / 16000
        assert any(lo - 0.1 <= m_sec <= hi + 0.1 for lo, hi in silent_regions), (
            f"midpoint at {m_sec:.2f}s not inside any expected silent region"
        )


def test_find_silence_midpoints_ignores_short_dips():
    # A brief dip (well under _MIN_SILENCE_RUN_SEC) surrounded by a MUCH
    # longer real silence elsewhere should NOT itself be reported as a
    # split candidate -- only the genuinely long run should be -- since a
    # short dip is exactly the kind of transient that must not trigger a
    # mid-word/mid-phrase split.
    samples = np.concatenate([
        _tone(1.0, 0.5), _silence(0.05), _tone(1.0, 0.5), _silence(1.5), _tone(1.0, 0.5),
    ])
    midpoints = find_silence_midpoints(samples, sample_rate=16000)
    # every returned midpoint must fall inside the LONG silent region [2.05s, 3.55s],
    # never inside the short 50ms dip around 1.0s.
    for m in midpoints:
        m_sec = m / 16000
        assert not (0.9 <= m_sec <= 1.15), f"short dip at ~1.0s was incorrectly reported as a split point ({m_sec:.2f}s)"


def test_find_silence_midpoints_empty_for_short_clip():
    samples = _tone(0.01, 0.5)  # shorter than one fbank frame
    assert find_silence_midpoints(samples, sample_rate=16000) == []


def test_find_silence_midpoints_rejects_wrong_sample_rate():
    import pytest
    with pytest.raises(ValueError, match="16kHz"):
        find_silence_midpoints(_tone(1.0, 0.5, sample_rate=8000), sample_rate=8000)


def test_choose_split_points_empty_candidates_returns_empty():
    assert choose_intra_surah_split_points([], decode_chunk_len=48, n_chunks_total=1000) == []


def test_choose_split_points_too_short_recording_returns_empty():
    # n_chunks_total below 2*(warmup_chunks+1) can't fit even one valid split
    result = choose_intra_surah_split_points(
        [48 * 50], decode_chunk_len=48, n_chunks_total=50, warmup_chunks=100
    )
    assert result == []


def test_choose_split_points_filters_candidates_too_close_to_edges():
    n_chunks_total = 1000
    warmup = 100
    # candidate right at the very start/end can't get a full warm-up window
    near_start = 5 * 48
    near_end = (n_chunks_total - 5) * 48
    valid_middle = (n_chunks_total // 2) * 48
    result = choose_intra_surah_split_points(
        [near_start, near_end, valid_middle], decode_chunk_len=48,
        n_chunks_total=n_chunks_total, warmup_chunks=warmup,
    )
    assert result == [n_chunks_total // 2]


def test_choose_split_points_does_not_drop_valid_candidates_in_shorter_gaps():
    # Regression test for a real bug found in code review: the greedy
    # bisection loop used to only ever consider the SINGLE LONGEST gap
    # each iteration, and would give up (returning fewer splits than
    # exist) the instant that one longest gap had no valid candidate --
    # even when a perfectly valid, well-separated candidate remained in a
    # SHORTER gap. Two clustered candidates (3000, 2900) in a 10,000-chunk
    # recording: picking 3000 first leaves the longest remaining gap
    # [3000, 10000) (length 7000) with no candidate, while the shorter gap
    # [0, 3000) (length 3000) still comfortably fits 2900
    # (0+50 <= 2900 <= 3000-50) -- both must be chosen, not just one.
    n_chunks_total = 10000
    warmup = 50
    decode_chunk_len = 48
    candidates = [3000 * decode_chunk_len, 2900 * decode_chunk_len]
    result = choose_intra_surah_split_points(
        candidates, decode_chunk_len=decode_chunk_len,
        n_chunks_total=n_chunks_total, warmup_chunks=warmup,
    )
    assert sorted(result) == [2900, 3000], (
        f"expected both well-separated candidates to be chosen, got {result}"
    )


def test_choose_split_points_respects_max_splits():
    n_chunks_total = 2000
    warmup = 50
    candidates = [(i * n_chunks_total // 6) * 48 for i in range(1, 6)]
    result = choose_intra_surah_split_points(
        candidates, decode_chunk_len=48, n_chunks_total=n_chunks_total,
        warmup_chunks=warmup, max_splits=2,
    )
    assert len(result) == 2
    assert result == sorted(result)


def test_choose_split_points_are_sorted_and_within_bounds():
    n_chunks_total = 3000
    warmup = 80
    candidates = [(i * n_chunks_total // 8) * 48 for i in range(1, 8)]
    result = choose_intra_surah_split_points(
        candidates, decode_chunk_len=48, n_chunks_total=n_chunks_total, warmup_chunks=warmup,
    )
    assert result == sorted(result)
    assert all(0 < c < n_chunks_total for c in result)
    assert len(result) == len(set(result))  # no duplicate split points
