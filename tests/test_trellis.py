"""Unit tests for trellis.py's engine-agnostic shared primitives
(build_ext, frame_spans_from_path, avg_logprob_along_path) -- these were
previously only exercised transitively through viterbi/repeats integration
tests, never with a focused test of their own documented edge cases (found
missing in code review).
"""
import numpy as np

from quran_forced_align.trellis import avg_logprob_along_path, build_ext, frame_spans_from_path


def _reference_frame_spans(path, num_states):
    first_seen = np.full(num_states, -1, dtype=np.int64)
    last_seen = np.full(num_states, -1, dtype=np.int64)
    for t, s in enumerate(path):
        if first_seen[s] == -1:
            first_seen[s] = t
        last_seen[s] = t
    return first_seen, last_seen


def test_build_ext_interleaves_blanks_around_every_ref_id():
    ext = build_ext([5, 7, 5], blank_id=0)
    assert ext.tolist() == [0, 5, 0, 7, 0, 5, 0]
    assert ext.dtype == np.int64


def test_build_ext_empty_ref_ids_is_just_the_blank():
    ext = build_ext([], blank_id=9)
    assert ext.tolist() == [9]


def test_frame_spans_from_path_first_last_seen():
    # path visits state 0 at frames 0-1, state 1 at frame 2, state 2 at frames 3-4;
    # state 3 is never visited.
    path = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    first_seen, last_seen = frame_spans_from_path(path, num_states=4)
    assert first_seen.tolist() == [0, 2, 3, -1]
    assert last_seen.tolist() == [1, 2, 4, -1]


def test_frame_spans_from_path_single_frame():
    path = np.array([2], dtype=np.int64)
    first_seen, last_seen = frame_spans_from_path(path, num_states=3)
    assert first_seen.tolist() == [-1, -1, 0]
    assert last_seen.tolist() == [-1, -1, 0]


def test_frame_spans_from_path_matches_reference_loop_random():
    # The vectorized duplicate-index fancy-assignment version must be
    # identical to the former Python loop on random monotonic paths with
    # repeats and never-occupied states.
    rng = np.random.default_rng(2024)
    for _ in range(300):
        T = int(rng.integers(1, 80))
        num_states = int(rng.integers(1, 12))
        # monotonic nondecreasing path with repeats, some states skipped
        path = np.sort(rng.integers(0, num_states, size=T)).astype(np.int64)
        first_seen, last_seen = frame_spans_from_path(path, num_states)
        ref_first, ref_last = _reference_frame_spans(path, num_states)
        assert np.array_equal(first_seen, ref_first)
        assert np.array_equal(last_seen, ref_last)


def test_frame_spans_from_path_first_and_last_seen_semantics():
    # Direct check of the vectorized trick's edge semantics: state with a
    # single frame, state visited in the middle, state never visited.
    path = np.array([0, 1, 1, 0, 2, 0], dtype=np.int64)
    first_seen, last_seen = frame_spans_from_path(path, num_states=4)
    # state 0: frames 0, 3, 5 -> first 0, last 5
    # state 1: frames 1, 2 -> first 1, last 2
    # state 2: frame 4 -> first 4, last 4
    # state 3: never -> -1, -1
    assert first_seen.tolist() == [0, 1, 4, -1]
    assert last_seen.tolist() == [5, 2, 4, -1]


def test_avg_logprob_along_path_basic():
    # 3 frames, vocab size 4; ext/path pick out specific (frame, symbol) cells.
    log_probs = np.array([
        [-1.0, -2.0, -3.0, -4.0],
        [-5.0, -6.0, -7.0, -8.0],
        [-9.0, -10.0, -11.0, -12.0],
    ])
    ext = np.array([0, 1, 2], dtype=np.int64)  # state -> symbol id
    path = np.array([0, 1, 2], dtype=np.int64)  # frame t occupies state t
    # chosen symbols: frame0->ext[0]=0 -> -1.0; frame1->ext[1]=1 -> -6.0; frame2->ext[2]=2 -> -11.0
    result = avg_logprob_along_path(log_probs, ext, path, start_frame=0, end_frame=2)
    assert result == np.mean([-1.0, -6.0, -11.0])


def test_avg_logprob_along_path_restricted_window():
    log_probs = np.array([
        [-1.0, -2.0],
        [-3.0, -4.0],
        [-5.0, -6.0],
    ])
    ext = np.array([0, 1], dtype=np.int64)
    path = np.array([0, 1, 1], dtype=np.int64)
    # restrict to frames [1, 2]: symbols ext[1]=1 at frame1 (-4.0), ext[1]=1 at frame2 (-6.0)
    result = avg_logprob_along_path(log_probs, ext, path, start_frame=1, end_frame=2)
    assert result == np.mean([-4.0, -6.0])


def test_avg_logprob_along_path_invalid_span_returns_neg_inf():
    log_probs = np.zeros((5, 3))
    ext = np.array([0, 1], dtype=np.int64)
    path = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    # end_frame < start_frame: documented "fail closed" contract
    result = avg_logprob_along_path(log_probs, ext, path, start_frame=3, end_frame=1)
    assert result == -np.inf
