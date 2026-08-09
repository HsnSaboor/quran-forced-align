import numpy as np

# Below this many (T * M) trellis cells, the direct full-array
# implementation is used unconditionally -- it is faster (no chunking
# overhead) and its peak memory is already small at this scale. This is
# comfortably above the largest whole-surah case seen outside Al-Baqarah
# (surah 67: ~11,400 frames * ~2,557 states =~29M cells =~230MB float64)
# and every repeats.py local-window call (a few hundred frames * a few
# dozen states), while being far below what would ever risk exhausting
# memory (100M cells * 8 bytes = 800MB for `alpha` alone -- still trivial
# for any modern machine). Al-Baqarah's whole-surah call
# (~175,000 * ~49,097 =~8.6 BILLION cells) is ~90x over this threshold and
# unconditionally takes the checkpointed path.
_DIRECT_PATH_MAX_CELLS = 100_000_000


def _skip_valid_states(ext):
    """States s (odd, i.e. label states, s>=2) where the "skip a blank"
    transition (alpha[t-1, s-2] -> alpha[t, s]) is legal: CTC only allows
    skipping the blank between two DIFFERENT adjacent labels (two equal
    adjacent labels must have a real blank frame between them, or they'd
    collapse into one repeat under the standard CTC collapsing rule)."""
    M = len(ext)
    idxs = np.arange(2, M)
    mask = (idxs % 2 == 1) & (ext[idxs] != ext[idxs - 2])
    return idxs[mask]


def _skip_valid_states_and_set(ext):
    """Same as `_skip_valid_states`, but also returns it as a `set` for O(1)
    membership checks during backtrace -- both forms are needed at each of
    this module's two call sites (the array for `_step_alpha`'s vectorized
    indexing, the set for `_backtrack_step`'s scalar `in` check)."""
    skip_valid_states = _skip_valid_states(ext)
    skip_valid_set = set(skip_valid_states.tolist())
    return skip_valid_states, skip_valid_set


def _init_row0(log_probs, ext, M):
    """Row 0 (t=0) of the alpha trellis: only states 0 and 1 are reachable
    at the very first frame (there hasn't been time to advance further),
    so every other state starts at -inf. Shared by both the direct and
    checkpointed forward passes."""
    row0 = np.full(M, -np.inf, dtype=np.float64)
    row0[0] = log_probs[0, ext[0]]
    if M > 1:
        row0[1] = log_probs[0, ext[1]]
    return row0


def _step_alpha(alpha_prev, lp, ext, skip_valid_states, out, adv1_scratch, skip2_scratch):
    """One forward-recursion step: out[s] = max(stay, advance, skip) +
    lp[ext[s]], writing into caller-provided scratch buffers instead of
    allocating fresh arrays every call -- this function is invoked once per
    audio frame (up to ~175,000 times for the largest surah), so avoiding
    per-call allocation matters for both speed (no malloc/free churn) and
    the small transient memory otherwise incurred by np.concatenate/np.stack
    every iteration. `out` may not alias `alpha_prev` (the caller must
    already have `alpha_prev` written to storage this call won't overwrite
    before it's done reading it -- see the two call sites for exactly how
    each guarantees this).

    Numerically and in every branch decision this is IDENTICAL to computing
    `np.stack([stay, adv1, skip2]).max(axis=0) + lp[ext]` with `stay =
    alpha_prev`, `adv1 = shift(alpha_prev, 1)`, `skip2 = shift(alpha_prev,
    2)` restricted to legal skip states -- just without materializing those
    three arrays or the stacked array. `np.maximum` (binary, elementwise)
    chained twice associates the same max-of-three per element as
    `np.stack(...).max(axis=0)` (max is associative/commutative, and
    IEEE-754 max of non-NaN finite floats has no reassociation-order
    dependence the way sums do -- unlike the additive reductions
    pipeline.py's DETERMINISM section is careful about, elementwise max
    always picks the literal largest value regardless of comparison order).
    """
    # adv1_scratch[s] = alpha_prev[s-1] for s>=1, else -inf (no state -1).
    adv1_scratch[0] = -np.inf
    adv1_scratch[1:] = alpha_prev[:-1]
    # skip2_scratch[s] = alpha_prev[s-2] for legal skip states, else -inf
    # everywhere else. Reset only the (fixed, precomputed) legal positions
    # plus their complement once -- cheaper than np.full(M, -inf) every
    # call since skip_valid_states is typically a small fraction of M.
    skip2_scratch.fill(-np.inf)
    skip2_scratch[skip_valid_states] = alpha_prev[skip_valid_states - 2]

    np.maximum(alpha_prev, adv1_scratch, out=out)
    np.maximum(out, skip2_scratch, out=out)
    out += lp[ext]
    return out


def _backtrack_step(alpha_prev, s, skip_valid_set):
    """Recover which of the 3 candidates (stay=0, advance=1, skip=2)
    produced `alpha[t, s]`, given only `alpha[t-1]` (the emission term
    `lp[ext[s]]` that was added when `alpha[t, s]` was first computed is a
    constant additive offset shared by all 3 candidates for this (t, s), so
    it never affects which one is the argmax -- no need to pass it in).

    Checked in the SAME order (stay, advance, skip) that the forward pass's
    `np.maximum` chain effectively prioritizes on exact ties: `np.maximum`
    keeps its first argument on a tie (IEEE-754 max(a,a)=a either way, and
    for a tie between distinct-but-equal-valued predecessor branches this
    function's linear scan finds the first match exactly like the forward
    pass's left-to-right np.maximum chain would have folded it in) --
    matching np.argmax's own "first max wins" tie-break convention, so this
    reproduces the same choice the original stack+argmax formulation made,
    for the historical baseline this module verified byte-identical output
    against.

    Returns (predecessor_state, margin) where predecessor_state is s, s-1,
    or s-2, and `margin` is best_prev minus the second-best of the 3
    candidate values (+inf if only one candidate was finite -- there was no
    real alternative to disagree with). This margin is a genuinely free
    by-product of the backtrace: the 3 scalars were already being computed
    and compared to find the argmax; returning their spread as well adds no
    new pass, no new array, and no change to the chosen path. It answers
    the same qualitative question an ensemble of narrower beam-search
    widths would ("would a slightly different search have gone somewhere
    else here"), analytically rather than by materializing a second
    search -- see confidence.per_word_min_margin for how this is
    aggregated into a per-word low-confidence signal.
    """
    stay_val = alpha_prev[s]
    adv_val = alpha_prev[s - 1] if s >= 1 else -np.inf
    skip_val = alpha_prev[s - 2] if s >= 2 and s in skip_valid_set else -np.inf

    vals = sorted((stay_val, adv_val, skip_val), reverse=True)
    best_prev = vals[0]
    second_best = vals[1]
    margin = np.inf if not np.isfinite(second_best) else best_prev - second_best

    if stay_val == best_prev:
        return s, margin
    if adv_val == best_prev:
        return s - 1, margin
    return s - 2, margin


def _pick_end_state(alpha_last, M):
    """The Viterbi path must end in the final blank (M-1) or the final
    label (M-2, only reachable if M>=2) -- every other state at t=T-1
    cannot be the terminus of a valid CTC alignment (there'd be unconsumed
    reference tokens). Returns None if neither end candidate reached a
    valid (non -inf) score, meaning too few frames existed to fit this
    reference at all."""
    end_candidates = [M - 1]
    if M >= 2:
        end_candidates.append(M - 2)
    end_state = max(end_candidates, key=lambda s: alpha_last[s])
    if not np.isfinite(alpha_last[end_state]):
        return None
    return end_state
