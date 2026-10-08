"""Left-extension helper for _forward_echo_pass (stage 2 of gate-5).

Widens an accepted stage-1 repeat fix leftward and re-tests the widened
phrase against the SAME second-pass audio region. All stage-2 gates apply
(bilateral floor+1.0, per-word floor-0.75, onset <=25f, free-decode
cross-check rd>=0.40/margin>=-0.43, min durations). Best widening by score.
Only NEW (left-extension) words are appended as repeats.
"""

import numpy as np


def extend_fix_left(
    word_indices, cues, combined_token_ids, blank_id,
    min_word_dur_frames, confidence_floor,
    log_probs_np, seconds_per_frame, token_frame_spans_fn,
    spliced_cue_fn, fixed, consumed_idx, build_phrase_ids_fn,
):
    from ..decode import fast_ctc_align_c, greedy_ctc_decode_ids as _greedy
    from ..decode import token_id_levenshtein_ratio as _lev
    from ..trellis import avg_logprob_along_path

    T = log_probs_np.shape[0]
    j_first = min(word_indices)
    ext_words = []
    j = j_first - 1
    while j >= 0 and cues[j].get("aya") == cues[j_first].get("aya") and \
            cues[j].get("sura") == cues[j_first].get("sura"):
        ext_words.insert(0, j)
        j -= 1
        if len(ext_words) >= 8:
            break
    if not ext_words:
        return
    rep_cues = [c for c in fixed
                if c.get("is_repeat") and c.get("word") in
                set(cues[j].get("word") for j in word_indices)]
    if not rep_cues:
        return
    pad = int(0.5 / seconds_per_frame)
    rs = max(0, min(c["start_frame"] for c in rep_cues) - pad)
    re_ = min(T - 1, max(c["end_frame"] for c in rep_cues) + pad)
    if re_ - rs + 1 < 8:
        return
    win_lp = np.ascontiguousarray(log_probs_np[rs:re_ + 1], dtype=np.float32)

    def _positions_for(local_offset, count):
        return list(range(local_offset, local_offset + count))

    best = None
    max_try = len(ext_words)
    for n in range(1, max_try + 1):
        trial_words = ext_words[-n:] + list(word_indices)
        phrase_token_ids, _ = build_phrase_ids_fn(
            trial_words, cues, combined_token_ids)
        if not phrase_token_ids:
            continue
        ext1, path1, _ = fast_ctc_align_c(win_lp, phrase_token_ids, blank_id)
        if ext1 is None or path1 is None:
            continue
        nst = len(ext1)
        f_seen = np.full(nst, -1, dtype=np.int64)
        l_seen = np.full(nst, -1, dtype=np.int64)
        for t, st in enumerate(path1):
            st = int(st)
            if 0 <= st < nst:
                if f_seen[st] < 0:
                    f_seen[st] = t
                l_seen[st] = t
        Lstan = len(phrase_token_ids)
        need = np.arange(Lstan) * 2 + 1
        if np.any(f_seen[need] < 0) or np.any(l_seen[need] < 0):
            continue
        s0 = int(f_seen[1])
        e0 = int(l_seen[2 * Lstan - 1])
        if e0 <= s0:
            continue
        if s0 > 25:
            continue
        tok_spans = {}
        all_ok = True
        for m, j in enumerate(trial_words):
            nt = len(cues[j]["token_positions"])
            off = sum(len(cues[jj]["token_positions"])
                      for jj in trial_words[:m])
            ts, ss, se = token_frame_spans_fn(
                _positions_for(off, nt), f_seen, l_seen)
            if ss is None or se is None or ts is None:
                all_ok = False
                break
            tok_spans[j] = (ss, se, ts)
        if not all_ok:
            continue
        avg = avg_logprob_along_path(win_lp, ext1, path1, s0, e0)
        if avg < confidence_floor + 1.0:
            continue
        per_ok = True
        for j in trial_words:
            ss, se, _ = tok_spans[j]
            a = avg_logprob_along_path(win_lp, ext1, path1, ss, se)
            if a < confidence_floor - 0.75:
                per_ok = False
                break
        if not per_ok:
            continue
        dur = e0 - s0 + 1
        if dur < len(trial_words) * int(min_word_dur_frames):
            continue
        _dec = _greedy(win_lp, blank_id)
        _ph = list(phrase_token_ids)
        _db = _ph + _ph
        _rd = _lev(_dec, _db)
        _rs = _lev(_dec, _ph)
        if _rd < 0.40 or (_rd - _rs) < -0.43:
            continue
        _score = avg + (_rd - _rs) * 0.1
        if best is None or _score > best[0]:
            best = (_score, trial_words, tok_spans, ext1, path1, rs)
    if best is None:
        return
    _, trial_words, tok_spans, ext1, path1, win_start = best
    new_words = [j for j in trial_words if j not in word_indices]
    if not new_words:
        return
    for j in new_words:
        if j in consumed_idx:
            continue
        ss, se, ts = tok_spans[j]
        shifted = [(a + win_start, b + win_start) for (a, b) in ts]
        orig = cues[j]
        fixed.append(spliced_cue_fn(
            orig, ss + win_start, se + win_start, shifted,
            0, ext1, path1, True))
        fixed[-1]["repeat_stage"] = "forward-extend"
        consumed_idx.add(j)
