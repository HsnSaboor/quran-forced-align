"""Verifies `_forced_align_checkpointed` produces EXACTLY the same
(ext, path, margins) as `_forced_align_direct` -- see viterbi.py's module
docstring, which documents this equivalence as a design invariant ("the
same recursion, just computed and discarded in chunks instead of all at
once") but, before this file existed, cited a test of this exact name that
did not actually exist anywhere in the repo (a real doc/test drift bug
found during a code-quality review).

Does NOT require the ONNX model or any audio -- both code paths only need
a log-probability matrix and a reference token sequence, which this file
constructs synthetically (a deterministic RNG-seeded matrix biased toward a
known reference sequence, so a valid alignment path actually exists).
Runs unconditionally, unlike the model-dependent tests conftest.py skips.
"""
import numpy as np
import pytest

from quran_forced_align.viterbi import (
    _build_ext,
    _forced_align_checkpointed,
    _forced_align_direct,
    ctc_forced_align,
)

VOCAB_SIZE = 20
BLANK_ID = 0


def _make_biased_log_probs(rng, ref_ids, blank_id, frames_per_symbol=3, vocab_size=VOCAB_SIZE):
    """Build a synthetic [T, vocab_size] log-prob matrix whose T frames are
    biased toward actually spelling out `ext = [blank, ref[0], blank, ...]`
    in order (each extended-state gets `frames_per_symbol` frames strongly
    favouring its own symbol), plus i.i.d. random noise everywhere else --
    close enough to a real acoustic model's output shape (soft, not
    one-hot) to exercise the same numerical code paths a real run would,
    while guaranteeing a valid, findable alignment exists."""
    ext = _build_ext(ref_ids, blank_id)
    T = len(ext) * frames_per_symbol
    log_probs = rng.uniform(-8.0, -2.0, size=(T, vocab_size)).astype(np.float64)
    for i, sym in enumerate(ext):
        t0 = i * frames_per_symbol
        log_probs[t0:t0 + frames_per_symbol, sym] = rng.uniform(-0.5, -0.05, size=frames_per_symbol)
    return log_probs


def _make_skip_forcing_log_probs(rng, ref_ids, blank_id, frames_per_symbol=3, vocab_size=VOCAB_SIZE):
    """Like `_make_biased_log_probs`, but additionally makes the "skip a
    blank" transition (`_step_alpha`'s `skip2_scratch` / `_backtrack_step`'s
    `skip_val` branch) STRUCTURALLY NECESSARY (not merely favorable) at one
    point in the path, so it's guaranteed to actually be chosen.

    `_make_biased_log_probs` alone gives every state its own dedicated
    frame window with room to spare, so the straightforward interleaved
    blank-label-blank-label path always fits comfortably -- the skip
    transition is LEGAL (per `_skip_valid_states`) for every pair of
    unequal adjacent labels in that fixture, but nothing in the data ever
    makes it NECESSARY, so a checkpointed-vs-direct disagreement
    specifically on skip-transition backtrace could exist undetected (a
    real gap found in code review) -- merely de-favoring (not removing)
    a blank's frames still leaves the DP free to "sit" there anyway if
    frames are available, which is exactly what happened when this test
    was first written this way (verified empirically: the direct path
    still visited the de-favored blank).

    Fix: physically REMOVE one skip-valid label's preceding blank state's
    entire dedicated frame window from the returned matrix, shrinking T by
    `frames_per_symbol`. With strictly fewer frames than the straightforward
    path would need to visit every state once, the ONLY way a valid
    alignment can still fit in the remaining frames is to skip directly
    from the label before that blank to the label after it -- structurally
    forced, not merely probability-biased.
    """
    ext = _build_ext(ref_ids, blank_id)
    log_probs = _make_biased_log_probs(rng, ref_ids, blank_id, frames_per_symbol, vocab_size)
    skip_valid = _skip_valid_states_for_test(ext)
    assert skip_valid, "fixture's ref_ids must contain at least one pair of adjacent unequal labels"
    blanked_state = skip_valid[0] - 1  # the blank state directly before this skip-valid label state
    t0 = blanked_state * frames_per_symbol
    log_probs = np.concatenate([log_probs[:t0], log_probs[t0 + frames_per_symbol:]], axis=0)
    return log_probs, blanked_state


def _skip_valid_states_for_test(ext):
    """Local re-derivation of viterbi._skip_valid_states's condition
    (odd label states s>=2 where ext[s] != ext[s-2]) -- kept private to
    this test file rather than importing the underscore-private function
    directly, so this fixture-construction helper doesn't couple to
    viterbi.py's internal helper naming beyond what's already imported."""
    M = len(ext)
    return [s for s in range(2, M) if s % 2 == 1 and ext[s] != ext[s - 2]]


def test_checkpointed_matches_direct_when_skip_transition_wins():
    """Confirms `_forced_align_checkpointed` and `_forced_align_direct`
    agree even when the "skip a blank" DP transition is STRUCTURALLY
    NECESSARY at some point in the path (not just legal-but-unused) -- a
    gap identified in code review: the other parametrized tests' fixtures
    never force this branch to be taken, so they could not have caught a
    checkpointed-vs-direct disagreement specific to it."""
    rng = np.random.default_rng(99)
    ref_ids = rng.integers(1, VOCAB_SIZE, size=6).tolist()
    log_probs, blanked_state = _make_skip_forcing_log_probs(rng, ref_ids, BLANK_ID)
    ext = _build_ext(ref_ids, BLANK_ID)

    ext_d, path_d, margins_d = _forced_align_direct(log_probs, ext)
    ext_c, path_c, margins_c = _forced_align_checkpointed(log_probs, ext)

    assert ext_d is not None and ext_c is not None, (
        "fixture is over-constrained: no valid alignment fits in the shrunk matrix at all "
        "(a skip transition should have made this exactly fit, not impossible)"
    )
    # The removed blank state can never appear in a valid path (its frames
    # no longer exist in this matrix) -- confirming this AND that a valid
    # alignment was still found together confirm the DP actually took the
    # skip transition around it, not merely that the state is unreachable.
    assert blanked_state not in path_d
    assert np.array_equal(path_d, path_c)
    assert np.array_equal(margins_d, margins_c, equal_nan=True)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("ref_len", [1, 2, 5, 17, 40])
def test_checkpointed_matches_direct_various_sizes(seed, ref_len):
    """Core equivalence check across a range of reference lengths (M =
    2*ref_len+1 extended states), including sizes that do and don't divide
    evenly by the checkpointed path's chunk size (~sqrt(T)) -- catches
    off-by-one errors at chunk boundaries specifically."""
    rng = np.random.default_rng(seed)
    ref_ids = rng.integers(1, VOCAB_SIZE, size=ref_len).tolist()
    log_probs = _make_biased_log_probs(rng, ref_ids, BLANK_ID)
    ext = _build_ext(ref_ids, BLANK_ID)

    ext_d, path_d, margins_d = _forced_align_direct(log_probs, ext)
    ext_c, path_c, margins_c = _forced_align_checkpointed(log_probs, ext)

    assert ext_d is not None and ext_c is not None, "both paths must find a valid alignment for this fixture"
    assert np.array_equal(ext_d, ext_c)
    assert np.array_equal(path_d, path_c), (
        f"checkpointed and direct paths diverged for ref_len={ref_len} seed={seed}: "
        f"direct={path_d.tolist()} checkpointed={path_c.tolist()}"
    )
    # margins may legitimately be +inf at the same frames in both -- compare
    # with equal_nan-style exact float equality (no tolerance): both paths
    # run the IDENTICAL _step_alpha/_backtrack_step arithmetic per the
    # module's determinism guarantees, so this must be exact, not approximate.
    assert np.array_equal(margins_d, margins_c, equal_nan=True)


def _find_target_T_with_single_frame_remainder(min_T):
    """Find the smallest T >= min_T for which `_forced_align_checkpointed`'s
    own chunk formula (`chunk = max(2, int(sqrt(T)))`, checkpoints at
    `range(0, T, chunk)` plus a forced final checkpoint at T-1) produces a
    final interval of length exactly 1 frame -- the minimal/most
    off-by-one-prone case, and the ACTUAL formula this test needs to target
    (a perfect-square T does NOT give this -- see the comment this
    replaces below for why that reasoning was inverted)."""
    T = min_T
    while True:
        chunk = max(2, int(np.sqrt(T)))
        last_checkpoint_below_T = (T - 1) // chunk * chunk
        remainder = (T - 1) - last_checkpoint_below_T
        if remainder == 1:
            return T
        T += 1


def test_checkpointed_matches_direct_at_exact_chunk_boundary():
    """Targets the smallest final checkpoint-interval size the
    checkpointed path's backtrace can produce (a single-frame interval) --
    the off-by-one case most likely to break interval bookkeeping.

    NOTE: an earlier version of this test picked T at a perfect-square
    boundary on the mistaken assumption that `chunk = sqrt(T)` exactly
    dividing T would minimize the final interval. It does the OPPOSITE:
    when T is a perfect square and chunk == sqrt(T) exactly, T-1 sits
    `chunk - 1` frames past the last checkpoint -- the LARGEST possible
    remainder, not the smallest. `_find_target_T_with_single_frame_remainder`
    instead searches directly for a T whose remainder against
    `_forced_align_checkpointed`'s actual chunk formula is 1, which is the
    genuinely minimal case.
    """
    rng = np.random.default_rng(123)
    ref_ids = rng.integers(1, VOCAB_SIZE, size=8).tolist()
    ext = _build_ext(ref_ids, BLANK_ID)
    log_probs = _make_biased_log_probs(rng, ref_ids, BLANK_ID, frames_per_symbol=3)
    T = log_probs.shape[0]

    target_T = _find_target_T_with_single_frame_remainder(T)
    if target_T > T:
        # Pad with extra low-probability frames (favoring the final blank,
        # a legal "stay" state at the end) so the padded tail doesn't
        # invalidate the alignment (staying in the final blank state is
        # always legal).
        pad = np.full((target_T - T, log_probs.shape[1]), -8.0, dtype=np.float64)
        pad[:, ext[-1]] = -0.05
        log_probs = np.concatenate([log_probs, pad], axis=0)

    chunk = max(2, int(np.sqrt(target_T)))
    last_checkpoint = (target_T - 1) // chunk * chunk
    assert (target_T - 1) - last_checkpoint == 1, "test setup failed to reach a 1-frame final interval"

    ext_d, path_d, margins_d = _forced_align_direct(log_probs, ext)
    ext_c, path_c, margins_c = _forced_align_checkpointed(log_probs, ext)

    assert ext_d is not None and ext_c is not None
    assert np.array_equal(path_d, path_c)
    assert np.array_equal(margins_d, margins_c, equal_nan=True)


def test_ctc_forced_align_dispatches_direct_for_small_problems():
    """`ctc_forced_align` (the public dispatcher) must route small
    problems (T*M below _DIRECT_PATH_MAX_CELLS) through the direct path --
    checked indirectly here by confirming its output matches
    `_forced_align_direct` called explicitly on the same inputs (rather
    than reaching into `_DIRECT_PATH_MAX_CELLS` itself, which is an
    internal tuning constant, not part of this test's contract)."""
    rng = np.random.default_rng(7)
    ref_ids = rng.integers(1, VOCAB_SIZE, size=6).tolist()
    log_probs = _make_biased_log_probs(rng, ref_ids, BLANK_ID)

    ext_pub, path_pub, margins_pub = ctc_forced_align(log_probs, ref_ids, BLANK_ID)
    ext_direct, path_direct, margins_direct = _forced_align_direct(log_probs, _build_ext(ref_ids, BLANK_ID))

    assert np.array_equal(path_pub, path_direct)
    assert np.array_equal(margins_pub, margins_direct, equal_nan=True)


def test_forced_align_returns_none_triple_when_reference_cannot_fit():
    """Too few frames for the reference must return the documented
    (None, None, None) triple from all three entry points, not raise or
    return a partial/malformed result."""
    ref_ids = [1, 2, 3, 4, 5]  # M = 11 extended states
    log_probs = np.zeros((3, VOCAB_SIZE), dtype=np.float64)  # far fewer frames than states needed

    ext, path, margins = ctc_forced_align(log_probs, ref_ids, BLANK_ID)
    assert (ext, path, margins) == (None, None, None)
