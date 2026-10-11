"""Right-extension helper for stage-1 (C-engine) accepted fixes.

Widens an accepted stage-1 repeat fix rightward and re-tests the widened
phrase (accepted words + extra right words) as a DOUBLED reference against
a widened window covering both passes. Rationale: the C engine anchors on
an anomalous cue and sweeps K leftward, so its winning K stops AT the
anomaly -- but the true repeated phrase can continue PAST the anchor into
following words whose first pass was ABSORBED into the stretched anchor
span (e.g. Ghailan 6:99: K=3 wins on w7-9 at anchor w9, while w10-11's
first pass [48162,48217] sits absorbed inside the same anchor span and
their second pass [48280,48324] follows it; the doubled w7-11 fit over
[48098,48324] is perfect: bilateral -0.00, rd=1.0, margin +0.5).

All stage-1 gates apply to the widened trial (bilateral floor, per-word
floor-0.75, agreement, free-decode rd>=0.40/margin>=0.05/0.15, timing,
copy1-containment at K=1, edge-ownership, self-match). Best widening by
score. Only NEW (right-extension) words are appended as repeats; the
accepted words keep their stage-1 copies. New copies take copy2 spans
from the widened doubled fit.
"""

import numpy as np


# Gate thresholds mirrored from the C engine (both sweeps) so the widened
# trial is judged by the identical bar, not a weaker one.
_FREE_DECODE_MIN_RATIO_DOUBLED = 0.40
_FREE_DECODE_MIN_MARGIN_K1 = 0.05
_FREE_DECODE_MIN_MARGIN_KN = 0.15
_WORD_AGREEMENT_MAX_DIFF = 1.0
_WORD_AGREEMENT_FLOOR_SLACK = 1.0


def _first_last_seen(path1, nst):
    f = np.full(nst, -1, dtype=np.int64)
    l = np.full(nst, -1, dtype=np.int64)
    for t, st in enumerate(path1):
        st = int(st)
        if 0 <= st < nst:
            if f[st] < 0:
                f[st] = t
            l[st] = t
    return f, l


def extend_fix_right(
    word_indices, cues, combined_token_ids, blank_id,
    min_word_dur_frames, confidence_floor,
    log_probs_np, seconds_per_frame, token_frame_spans_fn,
    spliced_cue_fn, fixed, consumed_idx, build_phrase_ids_fn,
    ext_main=None, path_main=None,
):
    """Rightward widening of an accepted stage-1 fix (see module docstring).

    word_indices: accepted cue indices (ascending). Tries appending 1..8
    following same-ayah cues and re-fits the widened phrase DOUBLED over
    the window [accepted-window-start, next-cue-after-last-extension-start).
    Only genuinely new words' copy2 spans are appended as repeats.
    """
    from ..decode import fast_ctc_align_c, greedy_ctc_decode_ids as _greedy
    from ..decode import token_id_levenshtein_ratio as _lev
    from ..trellis import avg_logprob_along_path

    T = log_probs_np.shape[0]
    V = log_probs_np.shape[1]
    j_last = max(word_indices)
    ext_words = []
    j = j_last + 1
    while j < len(cues) and cues[j].get("aya") == cues[j_last].get("aya") and \
            cues[j].get("sura") == cues[j_last].get("sura"):
        ext_words.append(j)
        j += 1
        if len(ext_words) >= 8:
            break
    if not ext_words:
        return
    # The accepted window start: earliest copy1 frame of the accepted fix.
    # Recover from the already-spliced copy1 (non-repeat) cues in fixed.
    acc_words = set(cues[j].get("word") for j in word_indices)
    copy1_starts = [c["start_frame"] for c in fixed
                    if not c.get("is_repeat") and c.get("word") in acc_words
                    and c.get("aya") == cues[j_last].get("aya")]
    if word_indices[0] > 0:
        win_start = cues[word_indices[0] - 1]["end_frame"] + 1
    else:
        win_start = max(0, (min(copy1_starts) if copy1_starts else 0) - 8)
    if win_start < 0:
        win_start = 0

    def _positions_for(local_offset, count):
        return list(range(local_offset, local_offset + count))

    best = None
    max_try = len(ext_words)
    for n in range(1, max_try + 1):
        trial_words = list(word_indices) + ext_words[:n]
        last = trial_words[-1]
        win_end = cues[last + 1]["start_frame"] - 1 if last + 1 < len(cues) \
            else min(T - 1, cues[last]["end_frame"] + int(0.5 / seconds_per_frame))
        if win_end <= win_start:
            continue
        phrase_token_ids, _ = build_phrase_ids_fn(
            trial_words, cues, combined_token_ids)
        if not phrase_token_ids:
            continue
        L = len(phrase_token_ids)
        win_lp = np.ascontiguousarray(
            log_probs_np[win_start:win_end + 1], dtype=np.float32)
        db = list(phrase_token_ids) + list(phrase_token_ids)
        ext1, path1, _ = fast_ctc_align_c(win_lp, db, blank_id)
        if ext1 is None or path1 is None:
            continue
        nst = len(ext1)
        f_seen, l_seen = _first_last_seen(path1, nst)
        need = np.arange(2 * L) * 2 + 1
        if np.any(f_seen[need] < 0) or np.any(l_seen[need] < 0):
            continue
        c1s, c1e = int(f_seen[1]), int(l_seen[2 * L - 1])
        c2s, c2e = int(f_seen[2 * L + 1]), int(l_seen[4 * L - 1])
        if c2s <= c1e:
            continue
        # K=1 copy1-containment veto (mirror C engine).
        if len(trial_words) == 1:
            i0 = trial_words[0]
            if cues[i0]["start_frame"] <= win_start + c2s and \
                    win_start + c2e <= cues[i0]["end_frame"]:
                continue
        # Bilateral gate.
        avg = avg_logprob_along_path(win_lp, ext1, path1, c1s, c1e)
        avg2 = avg_logprob_along_path(win_lp, ext1, path1, c2s, c2e)
        bilat = avg if avg < avg2 else avg2
        if bilat < confidence_floor:
            continue
        # Per-word gates (mirror C engine: floor-0.75 bilateral form).
        tok_spans = {}
        all_ok = True
        acc = 0
        for j in trial_words:
            nt = len(cues[j]["token_positions"])
            ts1, ss1, se1 = token_frame_spans_fn(
                _positions_for(acc, nt), f_seen, l_seen)
            ts2, ss2, se2 = token_frame_spans_fn(
                _positions_for(L + acc, nt), f_seen, l_seen)
            if ss1 is None or se1 is None or ss2 is None or se2 is None:
                all_ok = False
                break
            a1 = avg_logprob_along_path(win_lp, ext1, path1, ss1, se1)
            a2 = avg_logprob_along_path(win_lp, ext1, path1, ss2, se2)
            if min(a1, a2) < confidence_floor - 0.75:
                all_ok = False
                break
            if abs(a1 - a2) > _WORD_AGREEMENT_MAX_DIFF and \
                    min(a1, a2) < confidence_floor - _WORD_AGREEMENT_FLOOR_SLACK:
                all_ok = False
                break
            tok_spans[j] = (ss2, se2, ts2)
            acc += nt
        if not all_ok:
            continue
        # Edge-ownership veto (mirror C engine): no copy2 word with BOTH
        # edge pins disowned.
        edge_veto = False
        acc = 0
        for j in trial_words:
            nt = len(cues[j]["token_positions"])
            ff = int(f_seen[2 * (L + acc) + 1])
            ll = int(l_seen[2 * (L + acc + nt - 1) + 1])
            own_f = own_l = False
            if ff >= 0:
                tok = ext1[int(path1[ff])]
                blp = float(win_lp[ff, tok])
                if blp >= float(np.max(win_lp[ff])) - 1e-4:
                    own_f = True
            if ll >= 0:
                tok = ext1[int(path1[ll])]
                blp = float(win_lp[ll, tok])
                if blp >= float(np.max(win_lp[ll])) - 1e-4:
                    own_l = True
            if not own_f and not own_l:
                edge_veto = True
                break
            acc += nt
        if edge_veto:
            continue
        # Self-match veto (mirror C engine): copy2 words must not sit
        # inside a DIFFERENT same-ayah cue's span (absorbed-echo skip:
        # exempt copy2 words overlapping their OWN cue's first-pass span,
        # which is exactly the absorbed case this extension targets...
        # so instead require: copy2 must not be contained in a LATER
        # (non-trial) cue's span).
        sm_veto = False
        acc = 0
        trial_set = set(trial_words)
        for j in trial_words:
            nt = len(cues[j]["token_positions"])
            s2 = win_start + int(f_seen[2 * (L + acc) + 1])
            e2 = win_start + int(l_seen[2 * (L + acc + nt - 1) + 1])
            acc += nt
            for m in range(len(cues)):
                if m in trial_set:
                    continue
                if cues[m].get("aya") != cues[j].get("aya") or \
                        cues[m].get("sura") != cues[j].get("sura"):
                    continue
                if cues[m]["start_frame"] <= s2 and e2 <= cues[m]["end_frame"]:
                    sm_veto = True
                    break
            if sm_veto:
                break
        if sm_veto:
            continue
        # Duration gate.
        if (c2e - c2s + 1) < int(min_word_dur_frames) or \
                (c1e - c1s + 1) < int(min_word_dur_frames):
            continue
        # ABSORBED-FIRST-PASS FORM (6:99 calibration): the extension's
        # first pass is ABSORBED inside the anchor span, so the doubled
        # trial fit itself splits copy1 inside [win_start, acc_c2e]:
        # genuine extension words' copy1 pins land strictly BEFORE the
        # accepted copy2 block ends (99:w10-11 copy1 [48162,48217] inside
        # the accepted window, before accepted copy2 end 48275), while a
        # non-repeated extension word's copy1 is pushed to its main-pass
        # location at/after the accepted copy2 end (70:w25 copy1
        # [31652,31668] vs accepted copy2 end 31644; 99:w30 copy1
        # [48992,49029] vs 48987; 156:w12 copy1 [82169,82195] vs 82164).
        # Require every extension word's copy1 to END strictly before the
        # accepted copy2 block ends -- AND to fit well there: the
        # extension word's copy1 span average must clear
        # confidence_floor - 0.75 on its own (the per-word gate above
        # averages... no, it checks min(a1,a2) which the copy2 side can
        # rescue; here we demand the copy1 side ALONE clears the bar, so
        # a fit that parks a garbage copy1 on unrelated audio (ayah91
        # w16 copy1 at -0.83: وَ@-0.13 هُ@-0.02 وَ@-7.45 fitted onto
        # w15-tail audio) is rejected even though its copy2 is perfect).
        # (Replaces the disjointness draft: the fit always places copy2
        # in order, so copy2-disjointness could never fire; the
        # absorbed-vs-main-pass distinction lives in copy1 placement,
        # which the fit cannot fake -- a genuine first pass is acoustic
        # fact inside the anchor span.)
        acc_c2e = -1
        _a2 = 0
        for j in word_indices:
            _nt = len(cues[j]["token_positions"])
            _e = int(l_seen[2 * (L + _a2 + _nt - 1) + 1])
            if _e > acc_c2e:
                acc_c2e = _e
            _a2 += _nt
        _absorbed = True
        _a3 = sum(len(cues[j]["token_positions"]) for j in word_indices)
        for j in ext_words[:n]:
            _nt = len(cues[j]["token_positions"])
            _s = int(f_seen[2 * _a3 + 1])
            _e = int(l_seen[2 * (_a3 + _nt - 1) + 1])
            if _e >= acc_c2e:
                _absorbed = False
                break
            _ca = avg_logprob_along_path(win_lp, ext1, path1, _s, _e)
            if _ca < confidence_floor - 0.75:
                _absorbed = False
                break
            # COPY1 PEAK-SUPPORT (ayah91 w16 calibration): the absorbed
            # copy1 must ALSO own its token peaks -- every extension
            # token's fitted frame must be owned by the fitted token
            # (frame argmax == fitted token). A genuine absorbed first
            # pass emits the tokens in order (99:w10-11 6/6 owned); a
            # garbage fit parks pins on others' peaks (ayah91 w16 copy1:
            # وَ@-0.13 on a وَ-owned frame but هُ@-0.02 and وَ@-7.45 on
            # w15-tail frames owned by رَ/اااا/ءَ -- only 1/3 owned).
            # Same ownership test as the stage-1 edge veto, extended to
            # ALL extension-copy1 tokens (copy1 is short: absorbed span).
            _sup = True
            for _q in range(_nt):
                _st = 2 * (_a3 + _q) + 1
                _a = int(f_seen[_st])
                if _a < 0:
                    _sup = False
                    break
                _tok = ext1[int(path1[_a])]
                _blp = float(win_lp[_a, _tok])
                if _blp < float(np.max(win_lp[_a])) - 1e-4:
                    _sup = False
                    break
            if not _sup:
                _absorbed = False
                break
            _a3 += _nt
        if not _absorbed:
            continue
        # Free-decode cross-check (mirror C engine: identical bar).
        dec = _greedy(win_lp, blank_id)
        rd = _lev(dec, db)
        rs = _lev(dec, list(phrase_token_ids))
        mm = _FREE_DECODE_MIN_MARGIN_K1 if len(trial_words) == 1 \
            else _FREE_DECODE_MIN_MARGIN_KN
        if rd < _FREE_DECODE_MIN_RATIO_DOUBLED or (rd - rs) < mm:
            continue
        score = bilat + (rd - rs) * 0.1
        if best is None or score > best[0]:
            best = (score, trial_words, tok_spans, ext1, path1, win_start)
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
        fixed[-1]["repeat_stage"] = "stage1-extend-right"
        consumed_idx.add(j)
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
