import numpy as np

from ..confidence import per_word_min_margin
from ..constants import (
    DEFAULT_MAX_REPEAT_WINDOW_WORDS,
    FREE_DECODE_MIN_MARGIN,
    FREE_DECODE_MIN_RATIO_DOUBLED,
    GAP_ARTIFACT_MAX_FRAMES,
    GAP_ARTIFACT_MIN_MARGIN,
)
from ..decode import _collapse_ctc_ids, token_id_levenshtein_ratio
from ..trellis import avg_logprob_along_path
from .candidate import _repeat_window_candidate, build_phrase_ids


def detect_and_fix_repeats(
    engine,
    cues,
    log_probs,
    combined_token_ids,
    blank_id,
    ext,
    path,
    anomaly_low_ratio,
    anomaly_high_ratio,
    min_word_dur_frames,
    ayah_final_high_ratio_mult=1.5,
    confidence_margin=1.0,
    max_repeat_window_words=None,
    max_passes=1,
):
    if not cues:
        return cues
    
    current_cues = cues
    for p in range(max_passes):
        fixed_cues = _detect_and_fix_repeats_pass(
            engine,
            current_cues,
            log_probs,
            combined_token_ids,
            blank_id,
            ext,
            path,
            anomaly_low_ratio,
            anomaly_high_ratio,
            min_word_dur_frames,
            ayah_final_high_ratio_mult=ayah_final_high_ratio_mult,
            confidence_margin=confidence_margin,
            max_repeat_window_words=max_repeat_window_words,
        )
        if len(fixed_cues) == len(current_cues):
            break
        current_cues = fixed_cues
        
    # Scan inter-word pause gaps once on converged cues for abandoned/pause restarts
    return _scan_pause_gap_restarts(current_cues, log_probs, combined_token_ids, blank_id, min_word_dur_frames)


def _detect_and_fix_repeats_pass(
    engine,
    cues,
    log_probs,
    combined_token_ids,
    blank_id,
    ext,
    path,
    anomaly_low_ratio,
    anomaly_high_ratio,
    min_word_dur_frames,
    ayah_final_high_ratio_mult=1.5,
    confidence_margin=1.0,
    max_repeat_window_words=DEFAULT_MAX_REPEAT_WINDOW_WORDS,
):
    low_ratio = anomaly_low_ratio
    high_ratio = anomaly_high_ratio
    """Adapted from build_surah_srt.py's two-pass repeat-detection design:
    after the main alignment, scan word durations for anomalies (near-zero
    or absurdly long relative to the surah's median word duration) as the
    signature of a repeat being silently absorbed into a neighbouring
    word's span. For each flagged word, re-run forced alignment LOCALLY
    (only over the frame window between its immediate neighbours) against a
    DOUBLED reference restricted to that audio span. If the doubled
    alignment finds two well-separated, plausible-duration,
    ACOUSTICALLY-CONFIDENT occurrences, splice cues back in place of the
    original single cue (one copy per word, per occurrence -- one flagged
    is_repeat=True). Bounded and local: only touches flagged words, never
    re-runs Viterbi over more than a small local window.

    `ext`/`path` are the WHOLE-SURAH main-pass Viterbi outputs (same ones
    `extract_word_frame_spans` was built from) -- needed here only to
    compute the "normal word" acoustic-confidence baseline below.

    THREE FIXES applied on top of the original (structurally broken)
    design. The first two were confirmed necessary by independent
    verification on surah 67 (338 words, 34 falsely flagged repeats before
    that fix); the third (widened K-search) fixes a separate RECALL bug
    found afterwards via ground-truth testing (see
    ground_truth_test.py/ground_truth_debug.py in the original repo):

    1. AYAH-FINAL ANOMALY THRESHOLD (secondary safety net). Natural waqf
       (pause) lengthening at the end of an ayah is a known, expected,
       NON-repeat cause of anomalously long duration -- verification found
       21/34 (62%) of false positives were exactly the last word of their
       ayah, with pre-fix duration ~3.1-4.2x the surah median (comfortably
       inside the default high_ratio=3.0 cutoff). Ayah-final words (the
       `is_ayah_final` flag threaded through from build_combined_reference)
       get their high-duration cutoff scaled up by `ayah_final_high_ratio_mult`
       before being flagged as anomalous at all. This is NOT a blanket ban:
       a genuine hifz-repeat landing on an ayah-final word would still
       clear this relaxed bar (or an even longer one) and get exactly the
       same acoustic-confidence scrutiny as every other candidate below --
       it only raises the bar for entering the doubled-realignment path in
       the first place.

    2. ACOUSTIC-CONFIDENCE CHECK. Timing plausibility (non-overlapping
       copies, both >= MIN_WORD_DUR) is NOT evidence of a genuine repeat:
       given any long window and a doubled reference, CTC forced alignment
       will basically always find *some* two-way split satisfying that --
       there is no null hypothesis ("this is NOT a repeat") anywhere in the
       original check. Verification found 32/34 flagged pairs had an
       almost-exactly-constant ~0.04s ("primary"/"repeat") gap -- one output
       frame at 40ms/frame, i.e. exactly the CTC trellis's mandatory
       blank-frame separator between two adjacent copies of the same label,
       not a genuine acoustic pause between two independently-uttered
       instances of a repeated word. We now also require each copy's
       average per-frame log-likelihood along its own accepted path
       (`avg_logprob_along_path`, the same per-frame values the main
       Viterbi's `alpha` recursion sums) to not be substantially worse than
       `confidence_margin` below the surah's own "normal word" baseline
       (the median of this same quantity over words that were NOT flagged
       as anomalous in step 1). A path that only exists because the DP
       always finds *a* path through the trellis -- not because the model
       is actually confident there's a matching phoneme sequence there --
       shows up as a much lower (more negative) average here.

    3. WIDENED K-WORD-WINDOW SEARCH (recall fix). A real hifz-practice
       repeat is usually a PHRASE of several words, not a single word.
       When a K-word phrase repeats, the absorption effect described above
       dumps almost all of the "extra" duration onto whichever word in the
       phrase happens to trip the anomaly threshold -- confirmed by
       ground-truth testing: with a genuine 4-word repeat (words 3-6 of a
       7-word test phrase, each repeated), only the LAST word (word 6)
       showed an anomalous duration; words 3-5 each individually measured
       normal duration in the main pass (they only ever "saw" their own
       single utterance's worth of time). Trying to fix this by doubling
       ONLY the one anomalous word (the original design) can never recover
       the correct split when the true repeated unit spans multiple words:
       it forces a 4-word-worth-of-audio window to fit a 1-word doubled
       reference, which either (a) finds some timing-plausible but
       acoustically-nonsensical two-way split of that single word's
       {own utterance + trailing unrelated audio} -- ground-truth "test_B"
       -- or (b) finds a split that the acoustic-confidence gate correctly
       rejects because forcing 4 words' worth of audio through a 1-word
       reference produces a catastrophically low log-likelihood for at
       least one of the two forced "copies" -- ground-truth "test_C".
       The fix: for each anomalous word at index i, try candidate
       REPEATED-PHRASE hypotheses of increasing width K = 1 .. k_max, where
       the K-word candidate phrase is the K words ENDING at (and including)
       i -- i.e. words [i-K+1 .. i] -- since the absorption effect
       concentrates the extra duration onto the word right before the
       pipeline "catches up" to the next correctly-timed word, the repeated
       phrase is the K-word window immediately preceding and including the
       anomalous word, not words starting at i. (This matches the K=1
       window-construction direction already used by the original
       single-word doubling logic -- see the `window_start` computation
       below, which is the K=1 case of the same formula.) `k_max` is bounded
       NATURALLY by how many words remain in word i's own ayah (a real
       hifz-practice repeat is a phrase or a whole ayah, never spanning
       into a different ayah), optionally additionally capped by
       `max_repeat_window_words` if that's not None -- see its definition
       (DEFAULT_MAX_REPEAT_WINDOW_WORDS) for why the ayah bound alone is
       sufficient for correctness on phrase repeats of any length, with no
       arbitrary ceiling needed. Each K is evaluated with the SAME
       `_repeat_window_candidate` timing check and the SAME
       acoustic-confidence gate as every other candidate -- widening the
       search space of window sizes is what lets a correctly-sized
       hypothesis (e.g. K=4) pass the identical bar a badly-fit K=1
       hypothesis failed, without weakening the gate itself. Among all K
       that produce a timing-plausible, confidence-passing split, we keep
       the one with the HIGHEST minimum bilateral confidence (min(avg1,
       avg2) across the two copies), tie-broken toward the smallest K on
       (near-)ties -- i.e. we prefer whichever window size the acoustic
       model is most confident about, and only prefer a wider phrase over a
       narrower one when it's a strictly better acoustic fit, not merely
       because it's available. If no K in [1, k_max] produces a passing
       split, the word is left unflagged, exactly as before this fix.

    4. GAP-ARTIFACT SECONDARY REJECT (tightens Fix 2, does not replace it).
       Widening the K-search (Fix 3) reopened a narrow gap for exactly the
       failure mode Fix 2 was written to close: an independent verification
       of the widened search on surah 67 found ayah 8's K=2 split (words
       'ٱلْغَيْظِ'+'كُلَّمَآ') still cleared `confidence_floor` -- by only
       0.034 nats, i.e. noise-level -- while ALSO showing the exact
       `gap_frames<=1` mechanical-artifact signature described in Fix 2's
       docstring above (word 84's real word-final long-vowel elongation was
       split into a 0.16s "copy1" + a 2.4s "copy2" that just relocates the
       original anomalous duration, not a genuine second utterance). A
       second case in the same investigation (idx=202, 'صَـٰٓفَّـٰتٍۢ', K=1)
       showed the identical pattern (gap_frames=1, margin 0.42) and is
       rejected as a bonus, though it predates this fix's motivating case.
       Formally clearing the floor is not enough evidence on its own when
       the split ALSO looks mechanical: we reject a candidate K if its gap
       between copy1_end and copy2_start is at or below
       `GAP_ARTIFACT_MAX_FRAMES` (the smallest gap the trellis can produce
       at all between two adjacent copies) AND its margin above
       `confidence_floor` is below `GAP_ARTIFACT_MIN_MARGIN` -- i.e. it only
       barely cleared the gate. This is deliberately NOT a second
       independent hard gate: a 1-frame gap with a large confidence margin
       (there is no such case in the data gathered so far, but the logic
       must not assume there never will be) is left alone, since strong
       acoustic evidence should still be able to override a minimal-gap
       split on its own merits. Every CONFIRMED genuine catch (ayah 28's
       K=3 widening below, and ground-truth test_B/test_C's 4-word
       recoveries) has gap_frames >= 7 (>=0.28s) and clears the floor by
       >=0.91 nats, comfortably outside both thresholds, so this reject is
       never reached for them -- see GAP_ARTIFACT_MAX_FRAMES/
       GAP_ARTIFACT_MIN_MARGIN's own comments for the full calibration.

    5. FREE-DECODE CROSS-CHECK (independent-evidence gate, tightens Fixes 2
       and 4 further). Fixes 2 and 4 only ever ask "does forcing a doubled
       reference onto this window produce a *locally coherent* path?" --
       they never ask "does the audio actually contain the phrase twice,
       independent of what we're trying to fit?" That gap matters because
       forced alignment has no null hypothesis: it will find *some*
       plausible-looking doubled split even when the audio only contains
       the phrase once, if the model's phoneme confidence happens to stay
       high across the boundary into whatever comes next (real phonemes ARE
       there, just not in the doubled pattern being tested for).

       We decode the K-window's `log_probs` slice (`log_probs[window_start:
       window_end + 1]` -- the exact same slice `_repeat_window_candidate`
       passes to the doubled-reference forced alignment; no new audio decode
       or ONNX inference needed) with `decode.greedy_ctc_decode_ids` and
       compare the result, at token-id granularity, against the single-copy
       phrase (`phrase_token_ids`) and the doubled phrase (`doubled_ids`)
       using `decode.token_id_levenshtein_ratio`. A genuine two-occurrence
       window is required to look MORE like two copies than one, with real
       headroom, and to look reasonably like two copies on an absolute basis
       (ruling out a low-confidence decode that happens to edge out
       ratio_single by chance): reject unless
       `ratio_doubled >= FREE_DECODE_MIN_RATIO_DOUBLED` AND
       `ratio_doubled - ratio_single >= FREE_DECODE_MIN_MARGIN`.

       NOTE ON GATE ORDER (performance + correctness): this gate runs BEFORE
       the `_repeat_window_candidate` forced-alignment call in the K loop
       (it only needs the window slice and the phrase id lists, both of
       which are available without any engine call). Every gate in this
       loop is a pure boolean rejection -- the `best` selection and
       `consumed` update happen only after ALL gates pass -- so reordering
       the gates is behavior-preserving, and the free-decode gate is by far
       the cheaper one to run first: it is the single most expensive thing
       per K that survives to the post-alignment stage (an O(T) collapse
       plus two O(n*m) Levenshtein DPs over decoded lengths that can reach
       thousands of frames), but the forced-alignment kernel launch it
       short-circuits is a GPU round-trip that must run for EVERY K that
       reaches it. With the gate in front, the majority of K values (which
       would fail the free-decode check anyway) never pay for the alignment
       at all. The gate's two Levenshtein calls also use the `min_ratio`
       early-exit bound (see `decode.token_id_levenshtein_ratio`'s
       docstring), which provably skips the DP whenever the true ratio is
       guaranteed to fail its threshold, so the decision sequence is
       byte-identical to the un-ordered exact-DP formulation.

       IMPORTANT CALIBRATION NOTE: an initial hand-rolled verification
       script (run against a narrow +/-1.5s pad around only the flagged
       SECOND-copy window) suggested 3 surah-67/68/71 candidates were all
       false positives. Re-running the SAME free-decode check against the
       ACTUAL K-search window this function constructs (which -- correctly
       -- extends back far enough to include the first copy, several
       seconds earlier) overturned that verdict for 2 of the 3: surah 67
       aya 28 (K=3) and surah 71 aya 7 (K=4) are genuine repeats under this
       independent check (ratio_doubled 0.96 and 1.00, matching the
       ground-truth fixtures' >=0.92) -- consistent with Fix 4's docstring
       above, written earlier, which already cites "ayah 28's K=3 widening"
       as a confirmed genuine catch. Only surah 68 aya 26 (K=2) is a
       confirmed false positive (ratio_doubled 0.44, ratio_single 0.89 --
       the free decode looks MORE like one occurrence than two). See
       FREE_DECODE_MIN_RATIO_DOUBLED/FREE_DECODE_MIN_MARGIN's own comments
       in constants.py for the full per-case numbers this threshold was
       calibrated against, and the general lesson: calibrate against the
       actual code path that will run, not an approximation of it. Applied
       identically for every K, same as every other gate.
    """

    if not cues:
        return cues

    durations = np.array([c["end_frame"] - c["start_frame"] for c in cues], dtype=np.float64)
    median = float(np.median(durations)) if len(durations) else 0.0
    if median <= 0:
        return cues
        
    import ctypes
    from ..decode import get_fast_ops
    
    _fast_ops = get_fast_ops()
    if _fast_ops is None or not hasattr(_fast_ops, "fast_detect_and_fix_repeats_engine"):
        # Fallback to python (but it shouldn't happen)
        print("WARNING: C engine not found, repeat detection will fail!")
        return cues

    num_cues = len(cues)
    cue_starts = np.array([c["start_frame"] for c in cues], dtype=np.int32)
    cue_ends = np.array([c["end_frame"] for c in cues], dtype=np.int32)
    cue_ayas = np.array([c["aya"] for c in cues], dtype=np.int32)
    cue_suras = np.array([c["sura"] for c in cues], dtype=np.int32)
    cue_is_ayah_final = np.array([1 if c.get("is_ayah_final") else 0 for c in cues], dtype=np.int8)
    
    cue_token_offsets = np.zeros(num_cues, dtype=np.int32)
    cue_token_counts = np.zeros(num_cues, dtype=np.int32)
    
    for i, c in enumerate(cues):
        if c["token_positions"]:
            cue_token_offsets[i] = c["token_positions"][0]
            cue_token_counts[i] = len(c["token_positions"])
            
    combined_token_ids_arr = np.array(combined_token_ids, dtype=np.int32)
    
    # log_probs
    if hasattr(log_probs, "detach"):
        log_probs_np = log_probs.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        log_probs_np = np.ascontiguousarray(log_probs, dtype=np.float32)
        
    T, V = log_probs_np.shape
    full_greedy_ids = np.argmax(log_probs_np, axis=-1).astype(np.int32)
    
    # baseline confidence
    def high_cutoff(c): return high_ratio * (ayah_final_high_ratio_mult if c.get("is_ayah_final") else 1.0)
    def is_anomalous(c): 
        d = c["end_frame"] - c["start_frame"]
        return d < low_ratio * median or d > high_cutoff(c) * median
    
    path_logprobs = log_probs_np[np.arange(T), ext[path]]
    cum_logprobs = np.pad(np.cumsum(path_logprobs), (1, 0))
    def fast_avg_logprob(sf, ef): return float((cum_logprobs[ef + 1] - cum_logprobs[sf]) / (ef - sf + 1)) if ef >= sf else -np.inf
    
    normal_avgs = [fast_avg_logprob(c["start_frame"], c["end_frame"]) for c in cues if not is_anomalous(c)]
    baseline_avg = float(np.median(normal_avgs)) if normal_avgs else 0.0
    confidence_floor = baseline_avg - confidence_margin
    
    out_K = np.zeros(num_cues, dtype=np.int32)
    out_window_start = np.zeros(num_cues, dtype=np.int32)
    out_window_end = np.zeros(num_cues, dtype=np.int32)
    out_path_offsets = np.zeros(num_cues, dtype=np.int32)
    out_path_lengths = np.zeros(num_cues, dtype=np.int32)
    
    # 2*T should be plenty for disjoint repeat windows
    out_paths = np.zeros(T * 2, dtype=np.int32)
    
    _max_rw = max_repeat_window_words if max_repeat_window_words is not None else -1
    margin = 2
    
    total_paths_len = _fast_ops.fast_detect_and_fix_repeats_engine(
        num_cues,
        cue_starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        cue_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        cue_ayas.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        cue_suras.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        cue_is_ayah_final.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        cue_token_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        cue_token_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        combined_token_ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        len(combined_token_ids_arr),
        log_probs_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        T, V,
        full_greedy_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        blank_id,
        confidence_floor,
        FREE_DECODE_MIN_RATIO_DOUBLED,
        FREE_DECODE_MIN_MARGIN,
        GAP_ARTIFACT_MAX_FRAMES,
        GAP_ARTIFACT_MIN_MARGIN,
        int(min_word_dur_frames),
        _max_rw,
        margin,
        median,
        low_ratio,
        high_ratio,
        ayah_final_high_ratio_mult,
        out_K.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_window_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_window_end.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_path_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_path_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_paths.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    )
    
    from ..trellis import frame_spans_from_path
    from .spans import token_frame_spans
    
    def _shift_token_spans(token_spans, offset):
        return [(s + offset, e + offset) for s, e in token_spans]
        
    def _spliced_cue(orig, start_local, end_local, token_spans_local, window_start, ext2, path2, is_repeat):
        return {
            **orig,
            "start_frame": start_local + window_start,
            "end_frame": end_local + window_start,
            "is_repeat": is_repeat,
            "token_frame_spans": _shift_token_spans(token_spans_local, window_start),
            "avg_logprob": avg_logprob_along_path(
                log_probs_np[window_start:window_start + len(path2)], ext2, path2, start_local, end_local
            ),
            "min_decision_margin": per_word_min_margin(None, start_local, end_local),
        }

    consumed = set()
    accepted_fixes = {}
    for i in range(num_cues):
        K = out_K[i]
        if K > 0:
            window_start = out_window_start[i]
            window_end = out_window_end[i]
            offset = out_path_offsets[i]
            length = out_path_lengths[i]
            path2 = out_paths[offset:offset+length]
            
            word_indices = list(range(i - K + 1, i + 1))
            phrase_token_ids, doubled_ids = build_phrase_ids(word_indices, cues, combined_token_ids)
            
            ext2 = np.empty(2 * len(doubled_ids) + 1, dtype=np.int32)
            for s in range(len(doubled_ids)):
                ext2[2*s] = blank_id
                ext2[2*s+1] = doubled_ids[s]
            ext2[-1] = blank_id
            
            first_seen, last_seen = frame_spans_from_path(path2, len(ext2))
            
            def positions_for(local_offset, count): return list(range(local_offset, local_offset + count))
            
            word_ntoks = [len(cues[j]["token_positions"]) for j in word_indices]
            offsets_l = []
            acc = 0
            for nt in word_ntoks:
                offsets_l.append(acc)
                acc += nt
            L_single = acc
            
            per_word_copy1 = {}
            per_word_copy2 = {}
            for m, j in enumerate(word_indices):
                nt = word_ntoks[m]
                s1_spans = token_frame_spans(positions_for(offsets_l[m], nt), first_seen, last_seen)
                s2_spans = token_frame_spans(positions_for(L_single + offsets_l[m], nt), first_seen, last_seen)
                per_word_copy1[j] = (s1_spans[1], s1_spans[2], s1_spans[0])
                per_word_copy2[j] = (s2_spans[1], s2_spans[2], s2_spans[0])
                
            accepted_fixes[i] = (word_indices, per_word_copy1, per_word_copy2, window_start, ext2, path2)
            consumed.update(word_indices)
            
    fixed = [c for j, c in enumerate(cues) if j not in consumed]
    for word_indices, per_word_copy1, per_word_copy2, window_start, ext2, path2 in accepted_fixes.values():
        for j in word_indices:
            s1, e1, tok_spans1 = per_word_copy1[j]
            orig = cues[j]
            fixed.append(_spliced_cue(orig, s1, e1, tok_spans1, window_start, ext2, path2, is_repeat=False))
        for j in word_indices:
            s2, e2, tok_spans2 = per_word_copy2[j]
            orig = cues[j]
            fixed.append(_spliced_cue(orig, s2, e2, tok_spans2, window_start, ext2, path2, is_repeat=True))
            
    fixed.sort(key=lambda c: (c["start_frame"], c["end_frame"]))
    
    return fixed


def _scan_pause_gap_restarts(cues, log_probs, combined_token_ids, blank_id, min_word_dur_frames):
    """Scan inter-word pause gaps >= 1.0s (25 frames) on converged cues for spoken restarts / repeat phrases."""
    if len(cues) < 2:
        return cues
        
    fixed = list(cues)
    gap_repeats = []
    
    import numpy as np
    from ..decode import token_id_levenshtein_ratio
    from ..trellis import build_ext, frame_spans_from_path, avg_logprob_along_path
    from ..viterbi import ctc_forced_align
    from ..confidence import per_word_min_margin
    from .spans import token_frame_spans
    
    log_probs_np = log_probs.cpu().numpy() if hasattr(log_probs, "cpu") else log_probs
    full_greedy_ids = np.argmax(log_probs_np, axis=-1).astype(np.int32)
    
    def _shift_token_spans(token_spans, offset):
        return [(s + offset, e + offset) for s, e in token_spans]
        
    def _spliced_gap_cue(orig, start_local, end_local, token_spans_local, window_start, ext2, path2, is_repeat):
        return {
            **orig,
            "start_frame": start_local + window_start,
            "end_frame": end_local + window_start,
            "is_repeat": is_repeat,
            "token_frame_spans": _shift_token_spans(token_spans_local, window_start),
            "avg_logprob": avg_logprob_along_path(
                log_probs_np[window_start:window_start + len(path2)], ext2, path2, start_local, end_local
            ),
            "min_decision_margin": per_word_min_margin(None, start_local, end_local),
        }

    for k in range(len(fixed) - 1):
        c_curr = fixed[k]
        c_next = fixed[k + 1]
        if c_curr["aya"] == c_next["aya"] and c_curr["sura"] == c_next["sura"]:
            gap_start = c_curr["end_frame"] + 1
            gap_end = c_next["start_frame"] - 1
            gap_len = gap_end - gap_start + 1
            if gap_len >= 25: # Pause >= 1.0s
                gap_greedy = full_greedy_ids[gap_start:gap_end + 1]
                gap_toks = []
                gap_frames = []
                prev_g = blank_id
                for t_idx, gid in enumerate(gap_greedy):
                    if gid != blank_id and gid != prev_g:
                        gap_toks.append(int(gid))
                        gap_frames.append(gap_start + t_idx)
                    prev_g = gid
                    
                aya_words = [c for c in fixed[:k + 1] if c["aya"] == c_curr["aya"] and c["sura"] == c_curr["sura"]][-15:]
                best_match = None
                best_score = 0.0
                best_g_start = 0
                restart_frame_start = gap_start
                
                if len(gap_toks) >= 4:
                    for g_s in range(len(gap_toks)):
                        trailing = gap_toks[g_s:]
                        for w_s in range(len(aya_words)):
                            for w_e in range(w_s, min(len(aya_words), w_s + 10)):
                                phrase_cands = aya_words[w_s:w_e + 1]
                                phrase_tok_ids = []
                                for c in phrase_cands:
                                    phrase_tok_ids.extend([combined_token_ids[pos] for pos in c.get("token_positions", [])])
                                if not phrase_tok_ids:
                                    continue
                                min_p = min(4, len(phrase_tok_ids))
                                for prefix_len in range(min_p, len(phrase_tok_ids) + 1):
                                    target = phrase_tok_ids[:prefix_len]
                                    for g_len in range(min_p, min(len(trailing) + 1, len(target) + 3)):
                                        r = token_id_levenshtein_ratio(trailing[:g_len], target)
                                        score = r + 0.15 * prefix_len
                                        if r >= 0.75 and score > best_score:
                                            best_score = score
                                            best_match = (phrase_cands, phrase_tok_ids[:prefix_len])
                                            best_g_start = g_s
                    if best_match:
                        restart_frame_start = gap_frames[best_g_start]
                
                # If greedy tokens were sparse in pause gap, test acoustic alignment of preceding suffix phrases (longest first)
                from ..decode import fast_ctc_align_c
                if not best_match and aya_words:
                    for w_s in range(max(0, len(aya_words) - 5), len(aya_words)):
                        phrase_cands = aya_words[w_s:]
                        phrase_tok_ids = []
                        for c in phrase_cands:
                            phrase_tok_ids.extend([combined_token_ids[pos] for pos in c.get("token_positions", [])])
                        if not phrase_tok_ids:
                            continue
                        lp_gap = log_probs_np[gap_start:gap_end + 1]
                        ext_gap, path_gap, _ = fast_ctc_align_c(lp_gap, phrase_tok_ids, blank_id)
                        if path_gap is not None:
                            alp = avg_logprob_along_path(lp_gap, ext_gap, path_gap, 0, len(path_gap) - 1)
                            if alp >= -1.35: # Acoustic confidence threshold for repeated speech in pause
                                best_match = (phrase_cands, phrase_tok_ids)
                                restart_frame_start = gap_start
                                break
                                
                if best_match:
                    phrase_cands, match_tok_ids = best_match
                    
                    # Check if phrase_cands is preceded by a short particle/preposition in aya_words (e.g. min, fee, an, wa, etc.)
                    start_cand_idx = aya_words.index(phrase_cands[0]) if phrase_cands[0] in aya_words else -1
                    if start_cand_idx > 0:
                        prev_cand = aya_words[start_cand_idx - 1]
                        prev_toks = [combined_token_ids[pos] for pos in prev_cand.get("token_positions", [])]
                        if 0 < len(prev_toks) <= 2:
                            test_tok_ids = prev_toks + list(match_tok_ids)
                            lp_gap_test = log_probs_np[gap_start:gap_end + 1]
                            ext_t, path_t, _ = fast_ctc_align_c(lp_gap_test, test_tok_ids, blank_id)
                            if path_t is not None:
                                first_st, _ = frame_spans_from_path(path_t, len(ext_t))
                                if first_st[1] >= 0:
                                    phrase_cands = [prev_cand] + list(phrase_cands)
                                    match_tok_ids = test_tok_ids
                                    restart_frame_start = gap_start
                    n_m = len(match_tok_ids)
                    
                    lp_gap = log_probs_np[restart_frame_start:gap_end + 1]
                    ext_gap, path_gap, _ = fast_ctc_align_c(lp_gap, match_tok_ids, blank_id)
                    if path_gap is not None:
                        first_s, last_s = frame_spans_from_path(path_gap, len(ext_gap))
                        visited_tokens = sum(1 for s in range(n_m) if first_s[2 * s + 1] >= 0)
                        if visited_tokens >= max(2, int(0.70 * n_m)):
                            tok_offset = 0
                            word_raw_spans = []
                            for cand in phrase_cands:
                                cand_toks = [combined_token_ids[pos] for pos in cand.get("token_positions", [])]
                                n_cand = len(cand_toks)
                                if tok_offset < n_m:
                                    n_part = min(n_cand, n_m - tok_offset)
                                    sub_pos = list(range(tok_offset, tok_offset + n_part))
                                    s_spans = token_frame_spans(sub_pos, first_s, last_s)
                                    word_raw_spans.append((cand, s_spans[1], s_spans[2], s_spans[0], n_part, n_cand))
                                    tok_offset += n_part
                                else:
                                    break
                            
                            if word_raw_spans:
                                p_start = max(0, first_s[1] if first_s[1] >= 0 else 0)
                                prev_end = p_start - 1
                                for cand, s_loc, e_loc, tok_visited, n_part, n_cand in word_raw_spans:
                                    cur_s = s_loc if s_loc > prev_end else prev_end + 1
                                    cur_e = e_loc if e_loc >= cur_s + min_word_dur_frames - 1 else cur_s + min_word_dur_frames - 1
                                    prev_end = cur_e
                                    rep_cand = cand
                                    if n_part < n_cand:
                                        rep_cand = {
                                            **cand,
                                            "token_positions": cand["token_positions"][:n_part],
                                            "token_char_idx": cand.get("token_char_idx", [])[:n_part] if "token_char_idx" in cand else [],
                                        }
                                    rep_cue = _spliced_gap_cue(
                                        rep_cand, cur_s, cur_e, tok_visited, restart_frame_start, ext_gap, path_gap, is_repeat=True
                                    )
                                    gap_repeats.append(rep_cue)
    if gap_repeats:
        fixed.extend(gap_repeats)
        fixed.sort(key=lambda c: (c["start_frame"], c["end_frame"]))
        
    return fixed
