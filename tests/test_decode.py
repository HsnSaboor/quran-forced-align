"""Unit tests for decode.py's greedy-CTC collapse (vectorized vs reference
Python loop) and the token_id_levenshtein_ratio `min_ratio` early-exit
bound. The bound's contract is subtle and worth pinning down exactly:
when it trips, the returned value is NOT the true ratio (it is an upper
bound >= it), so the only thing the caller may do with it is compare
against a threshold -- these tests verify both that a tripped return is
always strictly below the threshold (so the threshold comparison is
identical to what the exact DP would produce) and that a non-tripped
return is bit-identical to the no-bound call.
"""
import numpy as np
import pytest

from quran_forced_align.decode import (
    _collapse_ctc_ids,
    greedy_ctc_decode_ids,
    token_id_levenshtein_ratio,
)


def _reference_greedy_decode(log_probs, blank_id):
    """Exact Python-loop reference for greedy_ctc_decode_ids' semantics:
    argmax per frame; emit a non-blank id iff it differs from the
    previously EMITTED non-blank id OR the previous frame was blank."""
    ids = np.argmax(log_probs, axis=-1)
    out = []
    prev = None
    for tid in ids:
        tid = int(tid)
        if tid == blank_id:
            prev = None
            continue
        if tid == prev:
            continue
        prev = tid
        out.append(tid)
    return out


def _random_log_probs(rng, T, V):
    # Random emission scores; argmax ties are avoided (all-ones noise adds
    # a tiny per-cell perturbation) so the decode is well-defined.
    return rng.uniform(0.0, 1.0, size=(T, V)) + 1e-6 * rng.uniform(0.0, 1.0, size=(T, V))


def test_greedy_decode_matches_reference_loop_random():
    rng = np.random.default_rng(12345)
    for _ in range(200):
        T = rng.integers(1, 60)
        V = rng.integers(2, 12)
        blank_id = int(rng.integers(0, V))
        log_probs = _random_log_probs(rng, T, V)
        assert greedy_ctc_decode_ids(log_probs, blank_id) == _reference_greedy_decode(log_probs, blank_id)


def test_greedy_decode_hand_checked_cases():
    # Frames crafted so argmax picks specific ids at specific frames.
    def lps(frames, V):
        arr = np.zeros((len(frames), V))
        for t, (i, v) in enumerate(frames):
            arr[t, i] = v
        return arr

    blank = 0
    # all blank
    assert greedy_ctc_decode_ids(lps([(0, 1.0)] * 5, 3), blank) == []
    # no blanks: consecutive duplicates collapsed
    assert greedy_ctc_decode_ids(lps([(1, 1.0), (1, 1.0), (2, 1.0), (2, 1.0), (2, 1.0)], 3), blank) == [1, 2]
    # same id repeated after a blank: TWO emissions
    assert greedy_ctc_decode_ids(lps([(1, 1.0), (0, 1.0), (1, 1.0)], 3), blank) == [1, 1]
    # first frame non-blank (no preceding frame at all)
    assert greedy_ctc_decode_ids(lps([(1, 1.0), (2, 1.0)], 3), blank) == [1, 2]
    # trailing blanks dropped
    assert greedy_ctc_decode_ids(lps([(1, 1.0), (0, 1.0), (0, 1.0)], 3), blank) == [1]


def test_greedy_decode_empty_input():
    assert greedy_ctc_decode_ids(np.zeros((0, 5)), 0) == []
    assert _collapse_ctc_ids(np.zeros(0, dtype=np.int64), 0) == []


def test_greedy_decode_returns_python_ints():
    log_probs = np.zeros((10, 4))
    log_probs[:, 3] = 1.0
    out = greedy_ctc_decode_ids(log_probs, 0)
    assert out == [3]
    assert all(type(x) is int for x in out)


def test_collapse_of_argmax_slice_equals_decode_of_slice():
    # The K-search argmax-reuse optimization: collapsing a SUFFIX of a
    # precomputed argmax over a wider window must equal decoding that
    # suffix's own slice directly.
    rng = np.random.default_rng(999)
    for _ in range(200):
        T = rng.integers(10, 120)
        V = rng.integers(2, 12)
        blank_id = int(rng.integers(0, V))
        log_probs = _random_log_probs(rng, T, V)
        ids = np.argmax(log_probs, axis=-1)
        lo = int(rng.integers(0, T))
        hi = int(rng.integers(lo, T))
        direct = greedy_ctc_decode_ids(log_probs[lo:hi + 1], blank_id)
        reused = _collapse_ctc_ids(ids[lo:hi + 1], blank_id)
        assert reused == direct
        # and the suffix-of-precomputed form used by detection.py
        suffix = _collapse_ctc_ids(ids[lo:], blank_id)
        assert suffix == greedy_ctc_decode_ids(log_probs[lo:], blank_id)


def _exact_ratio(a, b):
    return token_id_levenshtein_ratio(list(a), list(b))


def test_levenshtein_bound_never_overstates_threshold():
    # For any a, b, t: either the bound returns the exact DP value, or the
    # returned value is < t -- and in that case the TRUE ratio is also < t.
    rng = np.random.default_rng(777)
    for _ in range(500):
        n = int(rng.integers(0, 30))
        m = int(rng.integers(0, 30))
        V = int(rng.integers(1, 8))
        a = [int(x) for x in rng.integers(0, V, size=n)]
        b = [int(x) for x in rng.integers(0, V, size=m)]
        t = float(rng.uniform(0.0, 1.1))
        exact = _exact_ratio(a, b)
        got = token_id_levenshtein_ratio(a, b, min_ratio=t)
        if got < t:
            assert exact < t, (a, b, t, got, exact)
        else:
            assert got == exact, (a, b, t, got, exact)


def test_levenshtein_bound_length_ratio_identity():
    # The bound return, when tripped, is exactly min(n,m)/max(n,m) --
    # equality cases included (len_ratio == t must NOT trip, since the
    # caller's `>= t` comparisons need exactness there).
    rng = np.random.default_rng(42)
    for _ in range(200):
        n = int(rng.integers(0, 25))
        m = int(rng.integers(0, 25))
        a = [int(x) for x in rng.integers(0, 5, size=n)]
        b = [int(x) for x in rng.integers(0, 5, size=m)]
        if n == 0 and m == 0:
            continue
        t = float(min(n, m) / max(n, m, 1)) + 1e-9
        got = token_id_levenshtein_ratio(a, b, min_ratio=t)
        assert got < t
        assert got == min(n, m) / max(n, m, 1)
        # exact ratio is at most the bound, so it must also fail the gate
        assert _exact_ratio(a, b) < t


def test_levenshtein_bound_no_trip_returns_exact():
    assert token_id_levenshtein_ratio([1, 2], [1, 2, 3, 4], min_ratio=0.2) == _exact_ratio([1, 2], [1, 2, 3, 4])
    # both empty always 1.0 regardless of bound
    assert token_id_levenshtein_ratio([], [], min_ratio=0.99) == 1.0
    assert token_id_levenshtein_ratio([], [], min_ratio=None) == 1.0
    # min_ratio <= 0 never trips (length ratio >= 0)
    assert token_id_levenshtein_ratio([1], [1, 2, 3, 4, 5], min_ratio=0.0) == _exact_ratio([1], [1, 2, 3, 4, 5])


def test_free_decode_gate_decisions_identical_with_bounds():
    # The exact gate-decision sequence from detect_and_fix_repeats:
    # (ratio_doubled >= R2) and (ratio_doubled - ratio_single >= M).
    # With the min_ratio bounds the boolean decision must match the
    # unbounded exact-DP computation for every input triple.
    R2 = 0.75
    M = 0.15
    rng = np.random.default_rng(31337)
    for _ in range(300):
        n_d = int(rng.integers(0, 40))
        n_p = int(rng.integers(0, 20))
        V = int(rng.integers(1, 8))
        decoded = [int(x) for x in rng.integers(0, V, size=n_d)]
        phrase = [int(x) for x in rng.integers(0, V, size=n_p)]
        doubled = phrase + phrase

        exact_doubled = _exact_ratio(decoded, doubled)
        exact_single = _exact_ratio(decoded, phrase)
        exact_decision = (
            exact_doubled >= R2 and (exact_doubled - exact_single) >= M
        )

        bound_doubled = token_id_levenshtein_ratio(decoded, doubled, min_ratio=R2)
        bound_single = token_id_levenshtein_ratio(decoded, phrase, min_ratio=bound_doubled - M)
        bound_decision = (
            bound_doubled >= R2 and (bound_doubled - bound_single) >= M
        )
        assert bound_decision == exact_decision, (
            decoded, phrase, exact_doubled, exact_single, bound_doubled, bound_single
        )
