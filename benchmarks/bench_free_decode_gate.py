"""Micro-benchmark of the repeat-detection free-decode gate hot path
(detect_and_fix_repeats' per-K FIX 5 gate), comparing the OLD per-K cost
(Python-loop greedy decode + two exact O(n*m) Levenshtein DPs) against the
NEW vectorized/bounded pipeline (precomputed per-word argmax + vectorized
collapse + min_ratio early-exit Levenshtein bound).

Scenario sized to the Al-Baqara worst case: a long ayah window with T=3000
frames decodes to n~3000 ids (max case), the candidate phrase is 100
tokens (doubled reference m=200), and the K-loop runs 64 hypotheses for
one anomalous word -- most of which fail the free-decode gate and now
never reach the forced-alignment kernel launch at all.

The "total loop" numbers are computed as per-call cost x loop length (the
loop body is a straight sum of independent per-K costs; simulating all 64
full-DP iterations at n=3000,m=200 would take minutes for no extra
information). Run with `uv run python benchmarks/bench_free_decode_gate.py`.
"""
import time

import numpy as np

from quran_forced_align.decode import _collapse_ctc_ids, token_id_levenshtein_ratio

FREE_DECODE_MIN_RATIO_DOUBLED = 0.75
FREE_DECODE_MIN_MARGIN = 0.15

V = 251
BLANK = 0


# --- BEFORE: the old per-K hot path (inlined verbatim semantics) ---

def old_greedy_decode(log_probs, blank_id):
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


def old_levenshtein_ratio(a, b):
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 1.0
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
    return 1.0 - prev_row[m] / max(n, m, 1)


def old_per_k(log_probs, phrase_ids, doubled_ids):
    decoded = old_greedy_decode(log_probs, BLANK)
    ratio_doubled = old_levenshtein_ratio(decoded, doubled_ids)
    ratio_single = old_levenshtein_ratio(decoded, phrase_ids)
    return ratio_doubled, ratio_single


def new_per_k(ids_suffix, phrase_ids, doubled_ids):
    decoded = _collapse_ctc_ids(ids_suffix, BLANK)
    ratio_doubled = token_id_levenshtein_ratio(decoded, doubled_ids, min_ratio=FREE_DECODE_MIN_RATIO_DOUBLED)
    if ratio_doubled < FREE_DECODE_MIN_RATIO_DOUBLED:
        return ratio_doubled, None
    ratio_single = token_id_levenshtein_ratio(decoded, phrase_ids, min_ratio=ratio_doubled - FREE_DECODE_MIN_MARGIN)
    return ratio_doubled, ratio_single


def timeit(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    T = 3000
    log_probs = rng.uniform(0.0, 1.0, size=(T, V))
    log_probs[log_probs.argmax(axis=-1) == BLANK, BLANK] -= 0.5  # keep decode interesting
    phrase_ids = [int(x) for x in rng.integers(1, V, size=100)]
    doubled_ids = phrase_ids + phrase_ids
    ids = np.argmax(log_probs, axis=-1)

    # Sanity: outputs identical.
    assert new_per_k(ids, phrase_ids, doubled_ids)[0] < FREE_DECODE_MIN_RATIO_DOUBLED
    old_rd, old_rs = old_per_k(log_probs, phrase_ids, doubled_ids)
    assert old_rd < FREE_DECODE_MIN_RATIO_DOUBLED, "setup should fail the gate"
    assert new_per_k(ids, phrase_ids, doubled_ids)[1] is None

    K_LOOP = 64

    print(f"free-decode gate hot path, T={T}, decoded n~{len(ids)}, phrase m={len(phrase_ids)}, "
          f"doubled m={len(doubled_ids)}, K-loop={K_LOOP}")
    print("-" * 78)

    # 1. decode/collapse per-call
    t_old_decode = timeit(lambda: old_greedy_decode(log_probs, BLANK), reps=5)
    t_new_decode = timeit(lambda: _collapse_ctc_ids(ids, BLANK), reps=50)
    print(f"greedy decode per window   BEFORE {t_old_decode*1e3:9.3f} ms   AFTER {t_new_decode*1e3:9.3f} ms   "
          f"{t_old_decode/t_new_decode:7.1f}x")

    # 2. levenshtein per-call, bound-tripping case (the majority of K)
    def new_gate_only():
        decoded = _collapse_ctc_ids(ids, BLANK)
        return token_id_levenshtein_ratio(decoded, doubled_ids, min_ratio=FREE_DECODE_MIN_RATIO_DOUBLED)
    t_old_gate = timeit(lambda: old_per_k(log_probs, phrase_ids, doubled_ids), reps=2)
    t_new_gate = timeit(new_gate_only, reps=50)
    print(f"full gate (decode+2 DP)    BEFORE {t_old_gate*1e3:9.3f} ms   AFTER {t_new_gate*1e3:9.3f} ms   "
          f"{t_old_gate/t_new_gate:7.1f}x")

    # 3. total loop cost for one anomalous word
    t_old_loop = t_old_gate * K_LOOP
    t_argmax = timeit(lambda: np.argmax(log_probs, axis=-1), reps=20)
    t_new_loop = t_argmax + t_new_decode * K_LOOP
    print(f"one K-loop ({K_LOOP} K's)      BEFORE {t_old_loop:9.3f} s    AFTER {t_new_loop:9.3f} s    "
          f"{t_old_loop/t_new_loop:7.1f}x")

    # 4. bound-NOT-tripped case (decoded len ~ m): DP still runs, but the
    #    collapse is vectorized and the gate ran before any alignment.
    ids_near = np.array([(1 if i % 2 == 0 else 2) for i in range(190)], dtype=np.int64)
    t_old_dp = timeit(lambda: old_levenshtein_ratio([int(x) for x in ids_near], doubled_ids), reps=3)
    t_new_dp = timeit(lambda: token_id_levenshtein_ratio(ids_near.tolist(), doubled_ids, min_ratio=0.75), reps=3)
    assert t_new_dp < float("inf")
    print(f"DP needed (n=190,m=200)     BEFORE {t_old_dp*1e3:9.3f} ms   AFTER {t_new_dp*1e3:9.3f} ms   "
          f"{t_old_dp/t_new_dp:7.1f}x  (identical exact DP; bound not tripped)")


if __name__ == "__main__":
    main()
