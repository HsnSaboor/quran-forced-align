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
import ctypes
import os
from pathlib import Path
import numpy as np

# Load compiled C acceleration library if present
_fast_ops = None
try:
    so_path = Path(__file__).parent / "_fast_ops.so"
    if so_path.exists():
        _fast_ops = ctypes.CDLL(str(so_path))
        _fast_ops.fast_token_id_levenshtein_ratio.argtypes = [
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
            ctypes.c_double,
        ]
        _fast_ops.fast_token_id_levenshtein_ratio.restype = ctypes.c_double
        
        if hasattr(_fast_ops, "fast_ctc_forced_align"):
            _fast_ops.fast_ctc_forced_align.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32)
            ]
            _fast_ops.fast_ctc_forced_align.restype = ctypes.c_int

        if hasattr(_fast_ops, "fast_detect_and_fix_repeats_engine"):
            _fast_ops.fast_detect_and_fix_repeats_engine.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int8),
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                ctypes.c_float, ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_float, ctypes.c_float,
                ctypes.c_int, ctypes.c_int,
                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
                ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ]
            _fast_ops.fast_detect_and_fix_repeats_engine.restype = ctypes.c_int
except Exception:
    _fast_ops = None


def _collapse_ctc_ids(ids, blank_id):
    """Collapse an already-argmaxed per-frame token-id sequence `ids` to the
    greedy-CTC output list (consecutive duplicate labels collapsed, blanks
    dropped)."""
    if hasattr(ids, "size") and ids.size == 0:
        return []
    if len(ids) == 0:
        return []
    
    # Fast vectorized numpy
    if isinstance(ids, np.ndarray):
        is_blank = ids == blank_id
        prev_labels = np.empty_like(ids)
        prev_labels[0] = blank_id
        prev_labels[1:] = ids[:-1]
        prev_is_blank = np.empty_like(is_blank)
        prev_is_blank[0] = True
        prev_is_blank[1:] = is_blank[:-1]
        keep = (~is_blank) & (prev_is_blank | (ids != prev_labels))
        return ids[keep].tolist()
    else:
        out = []
        prev = blank_id
        for val in ids:
            if val != blank_id and val != prev:
                out.append(val)
            prev = val
        return out


def greedy_ctc_decode_ids(log_probs, blank_id):
    """Plain unconstrained greedy CTC decode: argmax per frame, collapse
    consecutive duplicate labels, drop blanks."""
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
    """Similarity ratio in [0, 1] between two sequences of token ids.
    Accelerated with native C extension when available, falling back to Python DP.
    """
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 1.0
    cutoff = -1.0 if min_ratio is None else float(min_ratio)
    
    if _fast_ops is not None:
        try:
            a_arr = (ctypes.c_int32 * n)(*a)
            b_arr = (ctypes.c_int32 * m)(*b)
            return float(_fast_ops.fast_token_id_levenshtein_ratio(a_arr, n, b_arr, m, cutoff))
        except Exception:
            pass

    if min_ratio is not None:
        len_ratio = min(n, m) / max(n, m, 1)
        if len_ratio < min_ratio:
            return len_ratio

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
