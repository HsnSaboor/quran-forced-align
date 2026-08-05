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

    Returns (ext, path) where path[t] is the best state at frame t (so the
    caller can recover, for every reference token, the first/last frame at
    which its state was occupied). Returns (None, None) if there are fewer
    frames than the minimum required (T < number of extended states needed
    to reach the end), meaning this reference can't possibly fit in this
    audio span.

    Dispatches internally between two byte-identical implementations
    depending on problem size -- see this module's docstring.
    """
    T = log_probs.shape[0]
    L = len(ref_ids)
    M = 2 * L + 1

    if T < 1 or M < 1:
        return None, None

    ext = _build_ext(ref_ids, blank_id)

    if T * M <= _DIRECT_PATH_MAX_CELLS:
        return _forced_align_direct(log_probs, ext)
    return _forced_align_checkpointed(log_probs, ext)


def _build_ext(ref_ids, blank_id):
    """Build the blank-interleaved extended state sequence
    [blank, ref[0], blank, ref[1], blank, ..., ref[L-1], blank]."""
    ext = [blank_id]
    for r in ref_ids:
        ext.append(r)
        ext.append(blank_id)
    return np.asarray(ext, dtype=np.int64)


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
    M = len(ext)
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

    Returns the predecessor state, i.e. s, s-1, or s-2.
    """
    stay_val = alpha_prev[s]
    adv_val = alpha_prev[s - 1] if s >= 1 else -np.inf
    skip_val = alpha_prev[s - 2] if s >= 2 and s in skip_valid_set else -np.inf

    best_prev = max(stay_val, adv_val, skip_val)
    if stay_val == best_prev:
        return s
    if adv_val == best_prev:
        return s - 1
    return s - 2


def _forced_align_direct(log_probs, ext):
    """Full (T, M) alpha array, kept alive for the whole forward pass so
    the backtrace can walk it directly. Used when T*M is small enough that
    this is cheap -- see _DIRECT_PATH_MAX_CELLS. No `backptr` array is
    stored; predecessor states are recovered on the fly during backtrace
    via `_backtrack_step` (see this module's docstring for why that's
    exact, not approximate, given deterministic arithmetic)."""
    T = log_probs.shape[0]
    M = len(ext)

    NEG_INF = -np.inf
    alpha = np.full((T, M), NEG_INF, dtype=np.float64)

    alpha[0] = _init_row0(log_probs, ext, M)

    skip_valid_states, skip_valid_set = _skip_valid_states_and_set(ext)
    adv1_scratch = np.empty(M, dtype=np.float64)
    skip2_scratch = np.empty(M, dtype=np.float64)

    for t in range(1, T):
        _step_alpha(alpha[t - 1], log_probs[t], ext, skip_valid_states, alpha[t],
                    adv1_scratch, skip2_scratch)

    end_state = _pick_end_state(alpha[T - 1], M)
    if end_state is None:
        return None, None

    path = np.zeros(T, dtype=np.int64)
    s = end_state
    path[T - 1] = s
    for t in range(T - 1, 0, -1):
        s = _backtrack_step(alpha[t - 1], s, skip_valid_set)
        path[t - 1] = s

    return ext, path


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


# ---------------------------------------------------------------------------
# Checkpointed (Hirschberg-style) forward pass + backtrace.
#
# Exact reformulation of _forced_align_direct that never holds more than a
# small number of full alpha ROWS in memory at once. Standard technique for
# any linear DP whose backtrace needs the whole trellis but whose memory
# footprint must stay sub-linear in T*M -- the same idea behind Hirschberg's
# algorithm for edit distance and "checkpointed" backprop/Viterbi for very
# long sequences.
#
# Idea: run the forward pass once, but only KEEP every `chunk` -th row
# ("checkpoints") instead of every row. To backtrace, walk backward one
# checkpoint-interval at a time: given the checkpoint row at the START of an
# interval, re-run the forward recursion locally across just that interval
# (a `chunk`-sized scratch, discarded once the interval's backtrace is done)
# to reconstruct the interval's own alpha rows, then backtrace through that
# reconstructed interval starting from the FIRST checkpoint's alpha row
# that lies at the state the outer backtrace arrived at.
#
# This does strictly more FORWARD arithmetic than the direct path (each
# frame's forward step runs twice: once during the initial checkpoint pass,
# once again during whichever interval's backtrace reconstruction touches
# it) but each floating-point operation is the EXACT SAME arithmetic in the
# EXACT SAME order as _forced_align_direct would have done for that frame,
# so the result is provably identical, not merely similar -- verified by
# tests/test_viterbi_checkpoint_equivalence.py.
# ---------------------------------------------------------------------------

def _forced_align_checkpointed(log_probs, ext):
    T = log_probs.shape[0]
    M = len(ext)

    skip_valid_states, skip_valid_set = _skip_valid_states_and_set(ext)

    # Chunk size ~ sqrt(T) balances "how many checkpoint rows we keep" (T /
    # chunk) against "how much local recomputation one backtrace interval
    # costs" (chunk rows) -- the standard Hirschberg/checkpointing trade-off,
    # giving O(sqrt(T) * M) peak memory and ~2x the forward FLOPs of the
    # direct path (one initial checkpoint-only pass + one local
    # reconstruction pass per interval, each interval visited exactly once
    # during backtrace). Clamped to >=2 so there's always at least one real
    # interval even for small T (shouldn't happen in practice since this
    # path is only reached for huge T, but keeps the chunking math valid at
    # any size rather than silently assuming T is huge).
    chunk = max(2, int(np.sqrt(T)))

    adv1_scratch = np.empty(M, dtype=np.float64)
    skip2_scratch = np.empty(M, dtype=np.float64)

    # Pass 1: forward recursion over the WHOLE sequence, keeping only every
    # `chunk`-th row (checkpoints[k] = alpha row at t = k*chunk). This is
    # the exact same per-frame arithmetic _forced_align_direct's loop does;
    # only which rows get RETAINED differs.
    checkpoint_ts = list(range(0, T, chunk))
    if checkpoint_ts[-1] != T - 1:
        checkpoint_ts.append(T - 1)  # always keep the final row -- backtrace starts there
    checkpoints = {}

    row_prev = _init_row0(log_probs, ext, M)
    if 0 in checkpoint_ts:
        checkpoints[0] = row_prev.copy()

    row_cur = np.empty(M, dtype=np.float64)
    for t in range(1, T):
        _step_alpha(row_prev, log_probs[t], ext, skip_valid_states, row_cur,
                    adv1_scratch, skip2_scratch)
        row_prev, row_cur = row_cur, row_prev  # swap buffers, no copy
        if t in checkpoint_ts:
            checkpoints[t] = row_prev.copy()

    end_state = _pick_end_state(row_prev, M)  # row_prev now holds alpha[T-1]
    if end_state is None:
        return None, None

    # Pass 2: backtrace interval-by-interval, latest to earliest. For each
    # interval (checkpoint_ts[k-1], checkpoint_ts[k]], reconstruct that
    # interval's alpha rows locally (starting from the checkpoint row at
    # checkpoint_ts[k-1], which is already exact) and backtrace through
    # them, then discard the reconstruction and move to the previous
    # interval.
    path = np.zeros(T, dtype=np.int64)
    s = end_state
    path[T - 1] = s

    for k in range(len(checkpoint_ts) - 1, 0, -1):
        t_start = checkpoint_ts[k - 1]
        t_end = checkpoint_ts[k]
        # Reconstruct alpha rows for t in (t_start, t_end] from the exact
        # checkpoint at t_start, using the identical _step_alpha calls the
        # initial pass used for these same frames.
        local_rows = [checkpoints[t_start]]
        row_prev_local = checkpoints[t_start]
        for t in range(t_start + 1, t_end + 1):
            row_cur_local = np.empty(M, dtype=np.float64)
            _step_alpha(row_prev_local, log_probs[t], ext, skip_valid_states, row_cur_local,
                        adv1_scratch, skip2_scratch)
            local_rows.append(row_cur_local)
            row_prev_local = row_cur_local

        # Backtrace within this interval: path[t_end] == s already (set by
        # the previous iteration, or the initial end_state for the very
        # last interval). Walk from t_end down to t_start+1, each step
        # needing local_rows[t - t_start - 1] as alpha[t-1].
        for t in range(t_end, t_start, -1):
            alpha_prev_local = local_rows[t - t_start - 1]
            s = _backtrack_step(alpha_prev_local, s, skip_valid_set)
            path[t - 1] = s
        # path[t_start] is now set (from the loop's last iteration writing
        # path[t-1] at t=t_start+1); s already reflects that value for the
        # next (earlier) interval to continue from.

    return ext, path


def frame_spans_from_path(path, num_states):
    """For each extended-trellis state, the first and last frame index at
    which the (monotonic, nondecreasing) Viterbi path occupied that state.
    -1 means the state was never occupied (shouldn't happen for label
    states in a successful alignment, since every label must be visited)."""
    first_seen = np.full(num_states, -1, dtype=np.int64)
    last_seen = np.full(num_states, -1, dtype=np.int64)
    for t, s in enumerate(path):
        if first_seen[s] == -1:
            first_seen[s] = t
        last_seen[s] = t
    return first_seen, last_seen


def avg_logprob_along_path(log_probs, ext, path, start_frame, end_frame):
    """Mean per-frame log P(chosen-symbol | frame) along the Viterbi path,
    restricted to [start_frame, end_frame] inclusive (frame indices local to
    whatever `log_probs`/`path` this came from -- caller is responsible for
    windowing consistently). This is the same quantity the main Viterbi's
    `alpha` recursion accumulates a *sum* of (see _step_alpha's `out +=
    lp[ext]`); we just re-derive the per-frame values along the
    already-decided path and average them, rather than duplicating the DP.
    Used as an acoustic-confidence check: a path that only exists because
    the DP always finds *some* path through a trellis (forced alignment has
    no null hypothesis) will tend to have a much lower (more negative)
    average than a path the model is actually confident about, since every
    frame's log-prob directly reflects how much probability mass the model
    put on that exact symbol at that exact frame.

    Returns -inf for an empty/invalid span (end_frame < start_frame) so
    comparisons against it always fail closed (treated as "no confidence").
    """
    if end_frame < start_frame:
        return -np.inf
    frames = np.arange(start_frame, end_frame + 1)
    symbols = ext[path[frames]]
    return float(np.mean(log_probs[frames, symbols]))
