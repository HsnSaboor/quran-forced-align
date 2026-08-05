"""CTC forced-alignment Viterbi (blank-interleaved trellis)."""
import numpy as np


def ctc_forced_align(log_probs, ref_ids, blank_id):
    """Standard CTC forced-alignment forward/Viterbi algorithm (the same
    algorithm behind torchaudio.functional.forced_align / "CTC
    segmentation"), implemented directly in numpy.

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
    which its state was occupied).  Returns (None, None) if there are fewer
    frames than the minimum required (T < number of extended states needed
    to reach the end), meaning this reference can't possibly fit in this
    audio span.
    """
    T = log_probs.shape[0]
    L = len(ref_ids)
    ext = [blank_id]
    for r in ref_ids:
        ext.append(r)
        ext.append(blank_id)
    ext = np.asarray(ext, dtype=np.int64)
    M = len(ext)

    if T < 1 or M < 1:
        return None, None

    NEG_INF = -1e15
    alpha = np.full((T, M), NEG_INF, dtype=np.float64)
    backptr = np.zeros((T, M), dtype=np.int8)  # 0=stay, 1=advance1, 2=skip2

    alpha[0, 0] = log_probs[0, ext[0]]
    if M > 1:
        alpha[0, 1] = log_probs[0, ext[1]]

    # Precompute which states are legally reachable via the "skip a blank"
    # transition: only odd (label) states s>=2 where ext[s] != ext[s-2]
    # (i.e. the current label differs from the previous label, so CTC
    # doesn't require an explicit blank frame between them).
    idxs = np.arange(2, M)
    skip_mask = (idxs % 2 == 1) & (ext[idxs] != ext[idxs - 2])
    skip_valid_states = idxs[skip_mask]

    for t in range(1, T):
        lp = log_probs[t]
        stay = alpha[t - 1]
        adv1 = np.concatenate(([NEG_INF], alpha[t - 1, :-1]))
        skip2 = np.full(M, NEG_INF, dtype=np.float64)
        skip2[skip_valid_states] = alpha[t - 1, skip_valid_states - 2]

        stacked = np.stack([stay, adv1, skip2], axis=0)
        arg = np.argmax(stacked, axis=0)
        best_prev = stacked[arg, np.arange(M)]
        alpha[t] = best_prev + lp[ext]
        backptr[t] = arg

    end_candidates = [M - 1]
    if M >= 2:
        end_candidates.append(M - 2)
    end_state = max(end_candidates, key=lambda s: alpha[T - 1, s])
    if alpha[T - 1, end_state] <= NEG_INF / 2:
        return None, None  # no valid path reached the end -- too few frames

    path = np.zeros(T, dtype=np.int64)
    s = end_state
    path[T - 1] = s
    for t in range(T - 1, 0, -1):
        mv = backptr[t, s]
        if mv == 1:
            s -= 1
        elif mv == 2:
            s -= 2
        path[t - 1] = s

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
    `alpha` recursion accumulates a *sum* of (see ctc_forced_align's
    `alpha[t] = best_prev + lp[ext]`); we just re-derive the per-frame
    values along the already-decided path and average them, rather than
    duplicating the DP. Used as an acoustic-confidence check: a path that
    only exists because the DP always finds *some* path through a trellis
    (forced alignment has no null hypothesis) will tend to have a much
    lower (more negative) average than a path the model is actually
    confident about, since every frame's log-prob directly reflects how
    much probability mass the model put on that exact symbol at that exact
    frame.

    Returns -inf for an empty/invalid span (end_frame < start_frame) so
    comparisons against it always fail closed (treated as "no confidence").
    """
    if end_frame < start_frame:
        return -np.inf
    frames = np.arange(start_frame, end_frame + 1)
    symbols = ext[path[frames]]
    return float(np.mean(log_probs[frames, symbols]))
