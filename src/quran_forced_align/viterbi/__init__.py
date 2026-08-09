"""CTC forced-alignment Viterbi (blank-interleaved trellis).

PERFORMANCE NOTE (checkpointed backtrace): the naive implementation of this
algorithm keeps a full (T, M) `alpha` array alive for the whole forward pass
so the backtrace at the end can walk back through it. For a single ayah or
a whole small-to-medium surah this is fine (a few hundred MB at most), but
for the largest surah (Al-Baqarah, ~175,000 frames x ~49,000 states) that
array alone would need ~69GB of float64 -- not remotely feasible on a
normal machine. `ctc_forced_align` below picks between two INTERNALLY
IDENTICAL forward-recursion implementations based on the problem's size:

  - `_forced_align_direct`: keeps the full (T, M) `alpha` array (the
    straightforward version), used whenever T*M is small enough that this
    is cheap (this is always true for repeats.py's local re-alignment
    windows, and for any surah small enough that memory was never the
    issue in the first place).
  - `_forced_align_checkpointed`: an exact (not approximate) Hirschberg/
    checkpointed-Viterbi reformulation that only ever keeps O(sqrt(T)*M)
    of `alpha` resident at once, recomputing short chunks of the forward
    pass on demand during backtrace. This produces EXACTLY the same
    (ext, path) as the direct version -- it is the same recursion, just
    computed and discarded in chunks instead of all at once -- verified by
    tests/test_viterbi_checkpoint_equivalence.py, which asserts byte-for-byte
    equality between both code paths across a range of sizes including ones
    that straddle the internal chunk boundaries.

Neither path stores a `backptr` array. `backptr[t,s]` only ever records
which of 3 candidates (stay/advance/skip-blank) produced `alpha[t,s]`'s
value, and DETERMINISTIC ARITHMETIC (fixed single-threaded numpy, per
pipeline.py's DETERMINISM section) means that value is exactly reproducible
from `alpha[t]`/`alpha[t-1]`/`log_probs[t]` alone: `_backtrack_step` below
recomputes it in O(1) per backtrace step by checking the same 3 candidates
in the same order argmax would have. This removes the second full (T, M)
array outright (an int8 array the same shape as `alpha`) with zero
correctness risk, since it's an algebraic identity given deterministic
arithmetic, not an approximation.
"""
from ..trellis import avg_logprob_along_path, build_ext, frame_spans_from_path
from .checkpointed import _forced_align_checkpointed
from .direct import _forced_align_direct
from .dp import _DIRECT_PATH_MAX_CELLS

__all__ = [
    "avg_logprob_along_path",
    "build_ext",
    "ctc_forced_align",
    "frame_spans_from_path",
]


def ctc_forced_align(log_probs, ref_ids, blank_id):
    """Standard CTC forced-alignment forward/Viterbi algorithm (the same
    algorithm behind torchaudio.functional.forced_align / "CTC
    segmentation").

    Given log_probs [T, V] and a reference token sequence ref_ids [L], build
    the blank-interleaved extended state sequence
        ext = [blank, ref[0], blank, ref[1], blank, ..., ref[L-1], blank]
    (length M = 2L+1) and find the best (max log-likelihood) monotonic path
    through it, one state per audio frame, via dynamic programming:
        alpha[t, s] = max(alpha[t-1, s],       # stay in state s
                          alpha[t-1, s-1],      # advance to next state
                          alpha[t-1, s-2])      # skip a blank (only legal
                                                 # between two DIFFERENT
                                                 # adjacent labels -- CTC
                                                 # requires a real blank
                                                 # frame to separate two
                                                 # equal adjacent labels,
                                                 # otherwise they'd collapse
                                                 # into one repeat)
                   + log_probs[t, ext[s]]

    Returns (ext, path, margins) where path[t] is the best state at frame t
    (so the caller can recover, for every reference token, the first/last
    frame at which its state was occupied) and margins[t] is the backtrace
    decision margin at frame t (best-minus-second-best of the 3 candidate
    predecessor scores that decided path[t-1] -- see `_backtrack_step`;
    margins[0] is always +inf since there is no backtrace step INTO frame
    0). Returns (None, None, None) if there are fewer frames than the
    minimum required (T < number of extended states needed to reach the
    end), meaning this reference can't possibly fit in this audio span.

    Dispatches internally between two byte-identical implementations
    depending on problem size -- see this module's docstring.
    """
    T = log_probs.shape[0]
    L = len(ref_ids)
    M = 2 * L + 1

    if T < 1 or M < 1:
        return None, None, None

    ext = build_ext(ref_ids, blank_id)

    if T * M <= _DIRECT_PATH_MAX_CELLS:
        return _forced_align_direct(log_probs, ext)
    return _forced_align_checkpointed(log_probs, ext)


# Private alias kept for this module's own internal tests
# (tests/test_viterbi_checkpoint_equivalence.py imports the underscore-
# private name directly to build fixtures without depending on this
# module's public re-export policy). Delegates to the shared
# engine-agnostic implementation in trellis.py -- see this module's
# `build_ext` re-export above for the public name every other consumer
# should use instead.
_build_ext = build_ext
