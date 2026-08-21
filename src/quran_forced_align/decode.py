"""Unconstrained (reference-free) greedy CTC decode + token-id-sequence
similarity, used as an INDEPENDENT cross-check signal for repeat detection
(see repeats.detect_and_fix_repeats).

Why this is needed: `_repeat_window_candidate`'s doubled-reference forced
alignment (viterbi.ctc_forced_align) is a biased hypothesis test -- it is
TOLD to fit two copies of the phrase and will always find *some* monotonic
path that does so, whether or not the audio actually contains the phrase
twice (see repeats.py's docstring points 2/4 for the mechanical-artifact
failure mode this already causes). A plain argmax-per-frame decode with no
such prior is a genuinely independent signal: it only produces the phrase's
tokens twice if the model's own free (unbiased) output actually contains
them twice.
"""
import numpy as np


def _collapse_ctc_ids(ids, blank_id):
    """Collapse an already-argmaxed per-frame token-id sequence `ids` to the
    greedy-CTC output list (consecutive duplicate labels collapsed, blanks
    dropped), using the same vectorized "is_new_occurrence" trick as
    `engines.cuda._aligned_labels_to_state_path`: a frame starts a NEW
    emitted label iff it is non-blank AND (the previous frame was blank OR
    its label differs from the previous frame's label). The blank sentinel
    at index -1 (frame -1 counts as blank) makes the t==0 case fall out
    automatically. Extracted from `greedy_ctc_decode_ids` so the
    repeat-detection K-search can reuse one argmax over the widest window
    and collapse each nested suffix independently (the collapse is a pure
    function of the id sequence, so collapsing a suffix of a precomputed
    argmax is identical to decoding the suffix itself)."""
    if ids.size == 0:
        return []
    is_blank = ids == blank_id
    prev_labels = np.empty_like(ids)
    prev_labels[0] = blank_id  # sentinel: frame -1 counts as blank
    prev_labels[1:] = ids[:-1]
    prev_is_blank = np.empty_like(is_blank)
    prev_is_blank[0] = True
    prev_is_blank[1:] = is_blank[:-1]
    keep = (~is_blank) & (prev_is_blank | (ids != prev_labels))
    return ids[keep].tolist()


def greedy_ctc_decode_ids(log_probs, blank_id):
    """Plain unconstrained greedy CTC decode: argmax per frame, collapse
    consecutive duplicate labels, drop blanks. Returns a list of token ids
    (NOT strings) -- no reference bias, no forced alignment, just the
    model's own free output. Used as an independent cross-check signal for
    repeat detection (see detect_and_fix_repeats)."""
    if hasattr(log_probs, "argmax"):
        try:
            import torch
            if isinstance(log_probs, torch.Tensor):
                ids_np = log_probs.argmax(dim=-1).cpu().numpy()
                return _collapse_ctc_ids(ids_np, blank_id)
        except ImportError:
            pass
    return _collapse_ctc_ids(np.argmax(log_probs, axis=-1), blank_id)


def token_id_levenshtein_ratio(a, b, min_ratio=None):
    """Similarity ratio in [0, 1] between two sequences of token ids, based
    on plain O(n*m) Levenshtein edit distance (insert/delete/substitute, all
    cost 1) computed at TOKEN-ID granularity (not characters -- tokens.txt
    entries are frequently multi-character phoneme clusters, so treating
    token ids as chars via e.g. chr(id) would conflate distinct tokens or
    break on ids outside a usable char range). Implemented locally rather
    than depending on the `Levenshtein` package: that package's own
    `ratio()` uses an INDEL-distance formula (1 - dist/(len1+len2), no
    substitution op), which is not the same quantity and would shift the
    calibrated thresholds below for no benefit -- and `Levenshtein` is only
    a transitive dependency of quran-transcript, not declared directly by
    this package. These phrase token sequences are short (a handful of
    tokens per word, a few words per phrase), so the O(n*m) DP table is
    trivial cost.

    ratio = 1 - distance / max(len(a), len(b), 1), so identical sequences
    (including both empty) score 1.0, and completely unrelated same-length
    sequences score >= 0 (distance capped at max(len(a), len(b))).

    `min_ratio`, if not None, enables a PROVABLE early-exit bound instead
    of the full O(n*m) DP. Levenshtein distance always satisfies
    d >= |n - m| (you need at least that many insertions/deletions to
    change the lengths), so the true ratio satisfies
    ratio = 1 - d/max(n,m) <= min(n,m)/max(n,m). If that upper bound is
    itself below `min_ratio`, the true ratio is certainly below it too, so
    the caller's threshold comparison (`ratio >= min_ratio`) would reject
    no matter what -- return the bound value instead of running the DP.
    The returned value is NOT the true ratio (it is >= it), so it must
    only ever be used for `>=`-vs-threshold comparisons where "provably
    below threshold" and "exactly below threshold" are equivalent --
    exactly how detect_and_fix_repeats' free-decode gate uses it. When the
    bound does not trip, the full DP runs and the result is bit-identical
    to the no-bound call.
    """
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 1.0
    if min_ratio is not None:
        len_ratio = min(n, m) / max(n, m, 1)
        if len_ratio < min_ratio:
            return len_ratio
    # Single rolling row -- classic space-optimized Levenshtein DP. No
    # set()/dict() iteration anywhere, so this is fully deterministic given
    # deterministic inputs (a, b are plain Python lists built in a fixed
    # order upstream).
    prev_row = list(range(m + 1))
    for i in range(1, n + 1):
        cur_row = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost_sub = prev_row[j - 1] + (0 if ai == b[j - 1] else 1)
            cost_del = prev_row[j] + 1
            cost_ins = cur_row[j - 1] + 1
            cur_row[j] = min(cost_sub, cost_del, cost_ins)
        prev_row = cur_row
    distance = prev_row[m]
    return 1.0 - distance / max(n, m, 1)
