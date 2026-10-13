"""Decayed-echo rescue: narrow rescue for ONE confirmed failure mode.

When a reciter repeats a phrase and the repeated pass is sung softer
than the first (hifz revision / echo), the acoustic model can map the
echo's soft-onset words to blank (-15..-22 nats) while still fitting
them in a doubled forced alignment with moderate confidence. Gate-5's
C sweep then rejects the window -- typically at the per-word agreement
bar (copies differ by >1.0 nat through onset decay alone) or the
copy2 edge-ownership veto (soft-onset edge tokens disowned) -- even
though the echo is real. The weakened posteriors also make the rescued
word's own greedy support unreadable, so no posterior-only fallback can
confirm it: confirmation must come from the WAVEFORM domain, where the
decayed onset is still present (see `echo_confirm.py`).

The rescue is deliberately NARROW so it cannot become a second,
weaker repeat detector:

* Entry requires a stage-1 ACCEPTED K=1 fix at the same anchor region
  (a crisp anchor the full gate chain already vetted), and only tries
  widening that fix LEFT by exactly one same-ayah word (the K=1 ->
  K=2 step the C sweep's K-selection dropped). K-selection itself
  (widest-wins / best-score) is untouched.
* The widened K=2 doubled fit must pass EVERY gate-5 gate EXCEPT the
  bilateral floor's onset-decay corner: freed rd/margin, per-word
  floor-0.75, timing, self-match, gap-artifact -- all at their
  production values, recomputed Python-side from the returned path.
  The bilateral floor admits ONE bounded excuse (see ONSET_EXCUSE_*
  in constants.py): a whole-copy2 miss <= 0.05 nats with the anchor
  word's copy2 tail crisp (>= floor + 0.30 on every non-onset token)
  -- the decayed-onset signature (6:156: miss 0.0084, anchor tail
  -0.09/-0.06), as opposed to an absent echo (6:152: miss ~9-12
  nats, best tail token -10.62).
* The two exempted gates are replaced, per word, by the audio-domain
  confirmer: BOTH words' fitted copy2 (echo) spans must DTW-match
  their own FIRST-PASS cue spans (cost <= ECHO_CONFIRM_MAX_COST),
  padded 40ms both sides. The confirmer compares each echo against
  ITSELF, so wrong-word audio absorbed into the window cannot
  confirm (6:156: wrong-word controls 2.82-3.35+).
* Byte-identical when disabled (default) or when no stage-1 K=1 fix
  exists at the anchor: the function returns without touching `fixed`.

Calibrated on Ghailan 6:156 (w10+w11 echo; anchor w11 K=1 accepted;
widened K=2 copy2-vs-own-cue: w10 DTW 1.59, w11 DTW 1.11).
"""

import numpy as np

from ..constants import SAMPLE_RATE as _SAMPLE_RATE
from ..decode import (
    fast_ctc_align_c,
    greedy_ctc_decode_ids as _greedy,
)
from ..decode import token_id_levenshtein_ratio as _lev
from ..trellis import avg_logprob_along_path
from .candidate import build_phrase_ids
from .echo_confirm import ECHO_CONFIRM_MAX_COST, dtw_window_norm_cost


def _first_last_seen(path, nst):
    f = np.full(nst, -1, dtype=np.int64)
    l = np.full(nst, -1, dtype=np.int64)
    for t, st in enumerate(path):
        st = int(st)
        if 0 <= st < nst:
            if f[st] < 0:
                f[st] = t
            l[st] = t
    return f, l


def rescue_decayed_echo_left(
    anchor_i,
    accepted_words,
    cues,
    combined_token_ids,
    blank_id,
    min_word_dur_frames,
    confidence_floor,
    log_probs_np,
    seconds_per_frame,
    token_frame_spans_fn,
    spliced_cue_fn,
    fixed,
    consumed_idx,
    samples=None,
):
    """Try widening a stage-1 accepted K=1 fix one word LEFT.

    `anchor_i`: cue index of the accepted K=1 anchor. `accepted_words`:
    the accepted word_indices ([anchor_i]). On success appends BOTH
    words' copy2 repeat cues (the widened K=2 split -- the broad K=1
    copy2 covered both echoes unsplit and is SUPERSEDED: the caller
    drops the anchor's stale K=1 copy2) and returns True; otherwise
    False, leaving `fixed` untouched. Copy1 spans are never re-cut:
    the absorbed first passes already sit correctly timed in the
    main-pass cues.
    """
    if samples is None:
        return False
    if list(accepted_words) != [anchor_i]:
        return False
    j_new = anchor_i - 1
    if j_new < 0:
        return False
    if cues[j_new].get("aya") != cues[anchor_i].get("aya") or \
            cues[j_new].get("sura") != cues[anchor_i].get("sura"):
        return False
    if j_new in consumed_idx:
        return False

    from ..constants import (
        FREE_DECODE_MIN_MARGIN,
        FREE_DECODE_MIN_RATIO_DOUBLED,
        GAP_ARTIFACT_MAX_FRAMES,
        GAP_ARTIFACT_MIN_MARGIN,
    )

    T = log_probs_np.shape[0]
    trial_words = [j_new, anchor_i]
    phrase_token_ids, _ = build_phrase_ids(
        trial_words, cues, combined_token_ids)
    if not phrase_token_ids:
        return False
    nt_new = len(cues[j_new]["token_positions"])
    nt_anch = len(cues[anchor_i]["token_positions"])
    if nt_new == 0 or nt_anch == 0:
        return False

    # Window: previous cue end + 1 .. next cue start - 1 (same geometry
    # as the C engine's K-window construction). NOTE: cue frames and
    # log-prob frames share coordinates ONLY for the full-utterance
    # call path (the harness-style region dump needs its offset added
    # back by the caller -- production always passes full-utterance
    # log_probs, so no offset is needed here).
    win_start = cues[j_new - 1]["end_frame"] + 1 if j_new > 0 \
        else max(0, cues[j_new]["start_frame"] - 8)
    win_end = cues[anchor_i + 1]["start_frame"] - 1 \
        if anchor_i + 1 < len(cues) else min(
            T - 1, cues[anchor_i]["end_frame"] + int(0.5 / seconds_per_frame))
    if win_end <= win_start:
        return False
    win_lp = np.ascontiguousarray(
        log_probs_np[win_start:win_end + 1], dtype=np.float32)
    L = len(phrase_token_ids)
    db = list(phrase_token_ids) + list(phrase_token_ids)

    ext1, path1, _ = fast_ctc_align_c(win_lp, db, blank_id)
    if ext1 is None or path1 is None:
        return False
    nst = len(ext1)
    f_seen, l_seen = _first_last_seen(path1, nst)
    need = np.arange(2 * L) * 2 + 1
    if np.any(f_seen[need] < 0) or np.any(l_seen[need] < 0):
        return False
    c1s, c1e = int(f_seen[1]), int(l_seen[2 * L - 1])
    c2s, c2e = int(f_seen[2 * L + 1]), int(l_seen[4 * L - 1])
    if c2s <= c1e:
        return False
    if (c2e - c2s + 1) < int(min_word_dur_frames) or \
            (c1e - c1s + 1) < int(min_word_dur_frames):
        return False
    # Copy1 must END at/before the anchor's own span end (the absorbed
    # case: copy1 sits on the first-pass audio, not downstream). The
    # absorbed new word's first pass then needs NO cue surgery: the
    # main pass already merged it into the anchor's stretched cue span
    # (6:156: the [3280.64,3281.56] w10 audio sits inside the w11 cue
    # [3281.76,3283.08]-ish stretched span). So the rescue emits ONLY
    # copy2 repeat cues, never a copy1 non-repeat (which would double
    # the already-correct original). Mirrors the forward pass, which
    # likewise appends is_repeat=True cues only.
    if win_start + c1e > cues[anchor_i]["end_frame"]:
        return False

    # Bilateral floor (same bar as the C gate) with a BOUNDED
    # onset-decay excuse (6:156): when the whole-copy2 average misses
    # the floor by only a hair (<= ONSET_EXCUSE_MAX_MISS nats) AND the
    # ANCHOR word's copy2 TAIL (all its tokens except the first) is
    # crisp (>= floor + ONSET_TAIL_CRISP_MARGIN), the miss is
    # attributable to crushed onset tokens on a real second utterance
    # -- the exact decayed-onset signature (6:156: copy2 avg -0.6089
    # vs floor -0.6005, miss 0.0084; anchor w11 tail -0.09/-0.06) --
    # rather than an absent echo (6:152: miss ~9-12 nats, best tail
    # token -10.62: excluded by BOTH bounds). Only the anchor's tail
    # is consulted: the anchor is the stage-1-vetted word, and the new
    # (left) word's echo is carried independently by the phrase-level
    # freed gate, the self-match veto, and its OWN DTW confirmation
    # (6:156 w10: DTW 1.60-1.93 stable, nearest negative 3.38). The
    # excuse waives this ONE gate only: freed, self-match,
    # gap-artifact, per-word floor-0.75, and both words' DTW
    # confirmation still apply at full production bars. Killed-FP
    # windows (6:52/102/141/143/144) never reach here (no accepted K=1
    # anchor -> rescue not called); 6:136-style uniform crush dies on
    # the tail-crisp bound.
    avg1 = avg_logprob_along_path(win_lp, ext1, path1, c1s, c1e)
    avg2 = avg_logprob_along_path(win_lp, ext1, path1, c2s, c2e)
    if min(avg1, avg2) < confidence_floor:
        from ..constants import (
            ONSET_EXCUSE_MAX_MISS,
            ONSET_TAIL_CRISP_MARGIN,
        )
        _excused = False
        if min(avg1, avg2) >= confidence_floor - ONSET_EXCUSE_MAX_MISS \
                and nt_anch > 1:
            # Anchor tail tokens are copy2 token offsets
            # L + nt_new + 1 .. L + nt_new + nt_anch - 1. (Single-token
            # anchors have no tail: the excuse cannot apply -- nothing
            # distinguishes decay from absence.)
            _tail_ok = True
            for _k in range(1, nt_anch):
                _r = token_frame_spans_fn(
                    [L + nt_new + _k], f_seen, l_seen)
                if _r is None:
                    _tail_ok = False
                    break
                _ts, _ss, _se = _r
                _v = avg_logprob_along_path(
                    win_lp, ext1, path1, _ss, _se)
                if _v < confidence_floor + ONSET_TAIL_CRISP_MARGIN:
                    _tail_ok = False
                    break
            if _tail_ok:
                _excused = True
        if not _excused:
            return False

    # Per-word floor-0.75 (same bar; agreement bar EXEMPT -- onset
    # decay splits the copies, which is exactly the failure mode).
    def _pos(off, count):
        return list(range(off, off + count))

    per_spans = {}
    acc = 0
    for j, nt in ((j_new, nt_new), (anchor_i, nt_anch)):
        r1 = token_frame_spans_fn(
            _pos(acc, nt), f_seen, l_seen)
        r2 = token_frame_spans_fn(
            _pos(L + acc, nt), f_seen, l_seen)
        if r1 is None or r2 is None:
            return False
        ts1, ss1, se1 = r1
        ts2, ss2, se2 = r2
        # token_frame_spans returns (per_token_spans, word_start,
        # word_end) in LOCAL window coordinates.
        a1 = avg_logprob_along_path(win_lp, ext1, path1, ss1, se1)
        a2 = avg_logprob_along_path(win_lp, ext1, path1, ss2, se2)
        if min(a1, a2) < confidence_floor - 0.75:
            return False
        per_spans[j] = (ss1, se1, ts1, ss2, se2, ts2)
        acc += nt

    # Freed cross-check (same bar as K>=2 in the C gate).
    dec = _greedy(win_lp, blank_id)
    rd = _lev(dec, db)
    rs = _lev(dec, list(phrase_token_ids))
    if rd < FREE_DECODE_MIN_RATIO_DOUBLED or \
            (rd - rs) < FREE_DECODE_MIN_MARGIN:
        return False

    # Self-match veto (mirror C engine: copy2 must not sit inside a
    # DIFFERENT same-ayah cue's span; same absorbed-echo skip for the
    # immediate successor overlapping the anchor).
    trial_set = {j_new, anchor_i}
    for j in (j_new, anchor_i):
        ss2 = per_spans[j][3]
        se2 = per_spans[j][4]
        s2 = win_start + ss2
        e2 = win_start + se2
        for m in range(len(cues)):
            if m in trial_set:
                continue
            if cues[m].get("aya") != cues[j].get("aya") or \
                    cues[m].get("sura") != cues[j].get("sura"):
                continue
            if cues[m]["start_frame"] <= s2 and e2 <= cues[m]["end_frame"]:
                if m == anchor_i + 1 and s2 <= cues[anchor_i]["end_frame"] \
                        and cues[j_new]["start_frame"] <= e2:
                    continue
                return False

    # Gap-artifact veto (same bar as the C gate).
    if (c2s - c1e) <= GAP_ARTIFACT_MAX_FRAMES and \
            min(avg1, avg2) - confidence_floor < GAP_ARTIFACT_MIN_MARGIN:
        return False

    # Audio-domain confirmation (REPLACES agreement + edge-ownership
    # for both words): each word's fitted copy2 must DTW-match its own
    # FIRST-PASS cue span. The copy1 fitted span is NOT used: fitted
    # token boundaries run +/-1 frame hot versus the cue's sample
    # rounding, and even a 1-frame (640-sample, 4-MFCC-frame) edge
    # asymmetry measurably inflates the cost (6:156: 1.66 -> 1.82).
    # Padding both spans symmetrically (40ms) absorbs the residual
    # boundary error; calibration (6:156): genuine 1.11-1.59, nearest
    # padded negative 2.82, fitted next-word 3.35.
    f2s = int(round(seconds_per_frame * _SAMPLE_RATE))
    _pad = int(round(0.04 * _SAMPLE_RATE))
    # Confirmation geometry: the ECHO (copy2) is compared against the
    # word's own first-pass cue span (NOT the fitted copy1 -- see
    # comment above), in cue/log-prob frame coordinates (the
    # confidence_floor baseline above indexes path log-probs with cue
    # spans directly); `samples` runs at _SAMPLE_RATE Hz with one
    # model frame per `seconds_per_frame` seconds.
    for j in (j_new, anchor_i):
        _, _, _, ss2, se2, _ = per_spans[j]
        g0 = cues[j]["start_frame"] * f2s
        g1 = (cues[j]["end_frame"] + 1) * f2s
        b0 = (win_start + ss2) * f2s
        b1 = (win_start + se2 + 1) * f2s
        a0, a1_ = max(0, g0 - _pad), g1 + _pad
        b0p, b1p = max(0, b0 - _pad), b1 + _pad
        if a1_ <= a0 or b1p <= b0p or b1p > len(samples):
            return False
        cost = dtw_window_norm_cost(samples[a0:a1_], samples[b0p:b1p])
        if cost > ECHO_CONFIRM_MAX_COST:
            return False

    # Accept: append BOTH words' copy2 (repeat) through the same splice
    # contract as stage 1. The widened K=2 fit supplies the second-pass
    # spans for the new word AND the anchor, which the broad K=1 fit
    # could not separate (6:156: K=1 copy2 [3283.36,3286.56] spans both
    # echoes; widened copy2 w10 [3283.68,3285.08] + w11 [3286.08,3286.56]).
    # The anchor's staged K=1 copy2 is SUPERSEDED by the narrower K=2
    # split (same word, same audio, coarser span -- keeping both emits
    # two overlapping second passes); the caller drops it on success
    # (see the Stage 1b block). `spliced_cue_fn` takes LOCAL coordinates
    # + the window's own (ext1, path1) and shifts spans itself.
    for j in (j_new, anchor_i):
        _, _, _, ss2, se2, ts2 = per_spans[j]
        orig = cues[j]
        fixed.append(spliced_cue_fn(
            orig, ss2, se2, ts2,
            win_start, ext1, path1, True))
        fixed[-1]["repeat_stage"] = "stage1-echo-rescue"
        consumed_idx.add(j)
    fixed.sort(key=lambda c: (c["start_frame"], c["end_frame"]))
    return True
