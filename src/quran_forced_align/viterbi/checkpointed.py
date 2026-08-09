import numpy as np

from .dp import _init_row0, _pick_end_state, _skip_valid_states_and_set, _step_alpha, _backtrack_step

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
    # `checkpoint_ts` (a list) is kept for pass 2's ORDERED interval walk
    # below, but membership tests in the hot per-frame loop just below need
    # O(1) lookup, not the O(len(checkpoint_ts)) == O(sqrt(T)) a list's `in`
    # would cost repeated T times (a real O(T*sqrt(T)) waste at Al-Baqarah
    # scale -- ~175,000 frames * ~420-element scans -- found in code
    # review): a separate set gives O(1) per check for the exact same
    # membership test, at the cost of one extra O(sqrt(T))-sized set build.
    checkpoint_t_set = set(checkpoint_ts)
    checkpoints = {}

    row_prev = _init_row0(log_probs, ext, M)
    if 0 in checkpoint_t_set:
        checkpoints[0] = row_prev.copy()

    row_cur = np.empty(M, dtype=np.float64)
    for t in range(1, T):
        _step_alpha(row_prev, log_probs[t], ext, skip_valid_states, row_cur,
                    adv1_scratch, skip2_scratch)
        row_prev, row_cur = row_cur, row_prev  # swap buffers, no copy
        if t in checkpoint_t_set:
            checkpoints[t] = row_prev.copy()

    end_state = _pick_end_state(row_prev, M)  # row_prev now holds alpha[T-1]
    if end_state is None:
        return None, None, None

    # Pass 2: backtrace interval-by-interval, latest to earliest. For each
    # interval (checkpoint_ts[k-1], checkpoint_ts[k]], reconstruct that
    # interval's alpha rows locally (starting from the checkpoint row at
    # checkpoint_ts[k-1], which is already exact) and backtrace through
    # them, then discard the reconstruction and move to the previous
    # interval.
    path = np.zeros(T, dtype=np.int64)
    margins = np.full(T, np.inf, dtype=np.float64)
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
            s, margin = _backtrack_step(alpha_prev_local, s, skip_valid_set)
            path[t - 1] = s
            margins[t] = margin
        # path[t_start] is now set (from the loop's last iteration writing
        # path[t-1] at t=t_start+1); s already reflects that value for the
        # next (earlier) interval to continue from.

    return ext, path, margins
