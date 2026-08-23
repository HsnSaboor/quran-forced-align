"""Engine-agnostic CTC blank-interleaved trellis utilities.

Every forced-alignment engine (CPU numpy Viterbi in `viterbi.py`, CUDA
torchaudio-backed in `engines/cuda.py`) produces the SAME three-array
contract -- `ext` (the blank-interleaved extended state sequence), `path`
(the best state at each audio frame), and `margins` (a per-frame
alignment-confidence signal) -- so every consumer downstream of forced
alignment (`repeats.py`, `confidence.py`, `pipeline.py`) can stay
completely engine-agnostic. This module holds the pure functions that
operate on that shared contract, independent of which engine produced it.
"""
import numpy as np


def build_ext(ref_ids, blank_id):
    """Build the blank-interleaved extended state sequence
    [blank, ref[0], blank, ref[1], blank, ..., ref[L-1], blank] (length
    M = 2L+1) that every forced-alignment engine aligns against. State
    index s is a blank state if s is even, a label state (ref_ids[(s-1)//2])
    if s is odd."""
    ext = [blank_id]
    for r in ref_ids:
        ext.append(r)
        ext.append(blank_id)
    return np.asarray(ext, dtype=np.int64)


def frame_spans_from_path(path, num_states):
    """For each extended-trellis state, the first and last frame index at
    which the (monotonic, nondecreasing) alignment path occupied that
    state. -1 means the state was never occupied (shouldn't happen for
    label states in a successful alignment, since every label must be
    visited).

    Vectorized via duplicate-index fancy assignment, which keeps the LAST
    assigned value per index: `last_seen[path] = t` assigns in ascending
    frame order, so each state ends up with its latest frame; `first_seen`
    assigns `t` in REVERSE frame order, so each state ends up with its
    earliest frame -- identical to the former Python loop, without the
    O(T) interpreter overhead.
    """
    t = np.arange(path.shape[0])
    last_seen = np.full(num_states, -1, dtype=np.int64)
    first_seen = np.full(num_states, -1, dtype=np.int64)
    last_seen[path] = t
    first_seen[path[::-1]] = t[::-1]
    return first_seen, last_seen


def avg_logprob_along_path(log_probs, ext, path, start_frame, end_frame):
    """Mean per-frame log P(chosen-symbol | frame) along the alignment
    path, restricted to [start_frame, end_frame] inclusive (frame indices
    local to whatever `log_probs`/`path` this came from -- caller is
    responsible for windowing consistently). Used as an
    acoustic-confidence check: a path that only exists because forced
    alignment always finds *some* path through the trellis (there is no
    null hypothesis) will tend to have a much lower (more negative)
    average than a path the model is actually confident about, since every
    frame's log-prob directly reflects how much probability mass the model
    put on that exact symbol at that exact frame.

    Returns -inf for an empty/invalid span (end_frame < start_frame) so
    comparisons against it always fail closed (treated as "no confidence").
    """
    if path is None or len(path) == 0 or log_probs is None or len(log_probs) == 0:
        return -np.inf
    start_frame = max(0, start_frame)
    max_f = min(len(path) - 1, len(log_probs) - 1)
    end_frame = min(max_f, end_frame)
    if end_frame < start_frame:
        return -np.inf
    if not isinstance(log_probs, np.ndarray) and hasattr(log_probs, "is_cuda"):
        import torch
        frames = torch.arange(start_frame, end_frame + 1, device=log_probs.device)
        symbols = ext[path[start_frame:end_frame + 1]]
        symbols_t = torch.as_tensor(symbols, dtype=torch.int64, device=log_probs.device)
        return float(log_probs[frames, symbols_t].mean().item())
    frames = np.arange(start_frame, end_frame + 1)
    symbols = ext[path[frames]]
    return float(np.mean(log_probs[frames, symbols]))

