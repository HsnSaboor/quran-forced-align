"""Unit tests for repeats.spans.token_frame_spans' vectorized
reimplementation: the numpy fancy-indexing version must keep the exact
return contract of the former Python loop (list of (int, int) tuples, or
None when any token's label state was never occupied / no tokens), with
word_start/word_end = min/max over the per-token spans.
"""
import numpy as np

from quran_forced_align.repeats.spans import token_frame_spans


def _reference_token_frame_spans(token_positions, first_seen, last_seen):
    per_token_spans = []
    for p in token_positions:
        s = 2 * p + 1
        start, end = first_seen[s], last_seen[s]
        if start < 0 or end < 0:
            return None
        per_token_spans.append((int(start), int(end)))
    if not per_token_spans:
        return None
    word_start = min(s for s, _ in per_token_spans)
    word_end = max(e for _, e in per_token_spans)
    return per_token_spans, word_start, word_end


def test_token_frame_spans_basic():
    first_seen = np.array([0, 2, 5, 7, 9, 11], dtype=np.int64)
    last_seen = np.array([1, 4, 6, 8, 10, 12], dtype=np.int64)
    # token 0 -> state 1 (frames 2-4), token 1 -> state 3 (frames 7-8),
    # token 2 -> state 5 (frames 11-12)
    spans, word_start, word_end = token_frame_spans([0, 1, 2], first_seen, last_seen)
    assert spans == [(2, 4), (7, 8), (11, 12)]
    assert word_start == 2
    assert word_end == 12
    assert all(type(s) is int and type(e) is int for s, e in spans)


def test_token_frame_spans_returns_none_when_state_never_seen():
    first_seen = np.array([0, 2, -1, -1], dtype=np.int64)
    last_seen = np.array([1, 4, -1, -1], dtype=np.int64)
    # token 1 lives at label state 3, which was never occupied
    assert token_frame_spans([0, 1], first_seen, last_seen) is None


def test_token_frame_spans_empty_positions_returns_none():
    first_seen = np.array([0, 2], dtype=np.int64)
    last_seen = np.array([1, 4], dtype=np.int64)
    assert token_frame_spans([], first_seen, last_seen) is None


def test_token_frame_spans_matches_reference_random():
    rng = np.random.default_rng(555)
    for _ in range(400):
        n_tokens = int(rng.integers(0, 12))
        # state 2p+1 must stay in-bounds: n_states >= 2, p <= (n_states-2)//2
        n_states = int(rng.integers(2, 30))
        first_seen = rng.integers(-1, 60, size=n_states).astype(np.int64)
        last_seen = rng.integers(-1, 60, size=n_states).astype(np.int64)
        max_p = (n_states - 2) // 2
        token_positions = [int(x) for x in rng.integers(0, max(1, max_p + 1), size=n_tokens)]
        got = token_frame_spans(token_positions, first_seen, last_seen)
        ref = _reference_token_frame_spans(token_positions, first_seen, last_seen)
        if ref is None:
            assert got is None
        else:
            assert got == ref
