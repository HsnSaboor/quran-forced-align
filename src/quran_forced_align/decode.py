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


def greedy_ctc_decode_ids(log_probs, blank_id):
    """Plain unconstrained greedy CTC decode: argmax per frame, collapse
    consecutive duplicate labels, drop blanks. Returns a list of token ids
    (NOT strings) -- no reference bias, no forced alignment, just the
    model's own free output. Used as an independent cross-check signal for
    repeat detection (see detect_and_fix_repeats)."""
    ids = np.argmax(log_probs, axis=-1)
    out = []
    prev = None
    for tid in ids:
        tid = int(tid)
        if tid == blank_id:
            prev = None
            continue
        if tid == prev:
            continue
        prev = tid
        out.append(tid)
    return out


def token_id_levenshtein_ratio(a, b):
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
    """
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 1.0
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
