import numpy as np

from .dp import _init_row0, _pick_end_state, _skip_valid_states_and_set, _step_alpha, _backtrack_step


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
        return None, None, None

    path = np.zeros(T, dtype=np.int64)
    margins = np.full(T, np.inf, dtype=np.float64)
    s = end_state
    path[T - 1] = s
    for t in range(T - 1, 0, -1):
        s, margin = _backtrack_step(alpha[t - 1], s, skip_valid_set)
        path[t - 1] = s
        margins[t] = margin

    return ext, path, margins
