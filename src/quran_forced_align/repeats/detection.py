import numpy as np

from ..confidence import per_word_min_margin
from ..constants import (
    DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
    DEFAULT_MAX_REPEAT_WINDOW_WORDS,
    DEFAULT_REPEAT_CONFIDENCE_MARGIN,
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
    ayah_final_high_ratio_mult=DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
    confidence_margin=DEFAULT_REPEAT_CONFIDENCE_MARGIN,
    max_repeat_window_words=None,
    max_passes=1,
    enable_phon_hint=False,
    seconds_per_frame=None,
    samples=None,
):
    """STEP-2 phonetic hint (gate-5 pause-vs-repeat disambiguation).

    `enable_phon_hint=False` (default) is byte-identical to the historical
    behavior: no hint is computed, cues are untouched, and the C engine
    receives the same `confidence_floor`. Pass `enable_phon_hint=True`
    (with the engine's `seconds_per_frame`, e.g. 0.04) to annotate each cue
    with `cues[i]["phon_hint"]` (the `repeats/phon_hint.py` result dict)
    and apply its bounded consumption (see `_detect_and_fix_repeats_pass`).
    """
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
            enable_phon_hint=enable_phon_hint,
            seconds_per_frame=seconds_per_frame,
            samples=samples,
        )
        if len(fixed_cues) == len(current_cues):
            break
        current_cues = fixed_cues
        
    return current_cues


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
    ayah_final_high_ratio_mult=DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
    confidence_margin=DEFAULT_REPEAT_CONFIDENCE_MARGIN,
    max_repeat_window_words=DEFAULT_MAX_REPEAT_WINDOW_WORDS,
    enable_phon_hint=False,
    seconds_per_frame=None,
    samples=None,
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
       shows up as a much lower (more negative) average here. The per-word
       form of this check (in `_fast_ops.c`, both gate-5 sweeps) additionally
       carries (a) a DURATION-ASYMMETRY EXEMPTION: a genuine second
       utterance sung faster than the first trips the 1.0-nat copy-agreement
       bar through tempo alone (adv_ayahx w5: 12 vs 9 frames, hi-lo 1.351),
       so the agreement bar applies only when the copies are
       duration-asymmetric (>4x -- the 6:70 force-split FP was 15x); and
       (b) a COPY2-CONTIGUITY VETO: the claimed second utterance must be
       one contiguous sung pass (no >25-frame inter-word leap inside
       copy2 -- the adv_triple K=6 chimera stitched passes 101 frames apart;
       genuine passes show <= ~12-frame breath gaps).

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

    # STEP-2 phonetic hint (default OFF -- byte-identical when disabled).
    # Computed Python-side AFTER confidence_floor and BEFORE the C call, per
    # STEP2_REPORT.md section 1. When enabled, each cue gets
    # `cues[i]["phon_hint"]` (the `repeats/phon_hint.py` result dict for that
    # word, with gap_after measured to the NEXT cue's start in ms and dur in
    # frames). The scalar `confidence_floor` passed to the C engine is
    # UNCHANGED -- the C engine (`_fast_ops.c`) takes no per-word input, so
    # the bounded bonus is consumed purely in the Python `accepted_fixes`
    # loop below (a hint-positive candidate whose margin above the floor is
    # tighter than PAUSAL_BONUS_NATS additional nats is left unflagged,
    # since the hint can only break near-ties, never rescue deep failures).
    phon_hints = None
    if enable_phon_hint:
        from .phon_hint import phon_hint

        spf = seconds_per_frame if seconds_per_frame is not None else 0.04
        phon_hints = []
        for i, c in enumerate(cues):
            if i + 1 < num_cues:
                gap_frames = cues[i + 1]["start_frame"] - c["end_frame"]
            else:
                gap_frames = 0
            gap_after_ms = max(0.0, float(gap_frames) * spf * 1000.0)
            dur_frames = c["end_frame"] - c["start_frame"]
            hint = phon_hint(
                c.get("word", ""),
                pause_after_ms=gap_after_ms,
                measured_dur=float(dur_frames),
                median_dur=float(median),
            )
            phon_hints.append(hint)
            c["phon_hint"] = hint

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
            "repeat_stage": "c-engine" if is_repeat else None,
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

            # STEP-2 hint consumption (only when enable_phon_hint=True):
            # the C engine accepted this K-window against its scalar
            # confidence_floor using WHOLE-window + per-word bilateral avgs
            # over `log_probs_np[window_start:...]` -- all recomputable
            # Python-side from the returned path. Recompute the window's
            # bilateral minimum (same definition as the C gate:
            # min(avg1, avg2) over the copy1/copy2 spans) and require
            # margin_above_floor >= hint_bonus for a hint-positive trigger
            # word. In other words the hint raises the effective floor by
            # at most PAUSAL_BONUS_NATS (0.15) for a waqf-likely trigger:
            # a candidate that cleared the C floor by LESS than the hint
            # bonus at a pause site is exactly the near-tie the hint is
            # allowed to break (STEP2_REPORT.md proposal option (b),
            # "floor minus bonus" applied as a post-reject since the C
            # entry point accepts only a scalar floor -- per-word floors
            # would require touching `_fast_ops.c`, which is forbidden).
            # Candidates clearing by MORE than the bonus, and all
            # hint-negative candidates, are unaffected.
            if phon_hints is not None:
                hint = phon_hints[i]
                if hint.get("pausal_likely"):
                    L = len(doubled_ids) // 2
                    n_states = len(ext2)
                    win_lp = log_probs_np[window_start:window_start + len(path2)]
                    c1_s = int(first_seen[1]) if n_states > 1 else -1
                    c1_e = int(last_seen[2 * L - 1]) if n_states > 2 * L - 1 else -1
                    c2_s = int(first_seen[2 * L + 1]) if n_states > 2 * L + 1 else -1
                    c2_e = int(last_seen[4 * L - 1]) if n_states > 4 * L - 1 else -1
                    avg1 = avg_logprob_along_path(win_lp, ext2, path2, c1_s, c1_e)
                    avg2 = avg_logprob_along_path(win_lp, ext2, path2, c2_s, c2_e)
                    bilateral = min(avg1, avg2)
                    margin_above_floor = bilateral - confidence_floor
                    if margin_above_floor < float(hint.get("bonus_nats", 0.0) or 0.0):
                        # Waqf-likely trigger with a near-tie margin: leave
                        # the word unflagged (do NOT splice a repeat).
                        continue

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

    # Stage 1b (decayed-echo rescue, default ON -- set QFA_DISABLE_ECHO_RESCUE
    # to restore the pre-rescue behavior): for each stage-1 ACCEPTED K=1 fix, try widening one word
    # LEFT (the K=1 -> K=2 step the C sweep's K-selection dropped when
    # the left word's echo onset decayed under blank). See
    # `repeats/echo_rescue.py`. Requires the raw waveform (`samples`);
    # without it this stage is skipped. Runs BEFORE the forward pass so
    # the forward stage's consumed_idx sees rescued words.
    import os as _os0
    if _os0.environ.get("QFA_DISABLE_ECHO_RESCUE", "") == "" and samples is not None:
        from .echo_rescue import rescue_decayed_echo_left
        _rescue_consumed = set(consumed)
        for _anchor_i in sorted(accepted_fixes):
            _w_idx = accepted_fixes[_anchor_i][0]
            if list(_w_idx) != [_anchor_i]:
                continue
            # The broad K=1 copy2 of the anchor spans BOTH echoes unsplit
            # (6:156: [3283.36,3286.56] vs the widened w10+w11 split); on a
            # successful rescue it is superseded by the two narrower K=2
            # copy2 spans. Snapshot its identity BEFORE the call so it can
            # be dropped from `fixed` (never by span proximity -- the stale
            # and superseding spans deliberately overlap). accepted_fixes
            # value layout: (word_indices, per_word_copy1, per_word_copy2,
            # window_start, ext2, path2); per_word_copy2[j] is
            # (start_local, end_local, token_spans).
            _stale_k1 = None
            _pw2 = accepted_fixes[_anchor_i][2].get(_anchor_i)
            if _pw2 is not None:
                _ws = accepted_fixes[_anchor_i][3]
                for _c in fixed:
                    if _c.get("is_repeat") and _c.get("repeat_stage") == "c-engine" \
                            and (_c["start_frame"], _c["end_frame"]) == (_pw2[0] + _ws, _pw2[1] + _ws):
                        _stale_k1 = _c
                        break
            if rescue_decayed_echo_left(
                _anchor_i, _w_idx, cues, combined_token_ids, blank_id,
                min_word_dur_frames, confidence_floor,
                log_probs_np, seconds_per_frame if seconds_per_frame is not None else 0.04,
                token_frame_spans, _spliced_cue,
                fixed, _rescue_consumed,
                samples,
            ) and _stale_k1 is not None:
                fixed.remove(_stale_k1)
        consumed.update(_rescue_consumed)

    # Stage 2 (forward-anchored echo search): the C-engine sweep above looks
    # BACKWARD from anomalous cues, but when a repeated second pass is fully
    # absorbed into stretched main-pass cues, no single cue is anomalous at
    # the right place and the backward window never covers the echo audio
    # (e.g. Ghailan 6:70 w17-24: second pass at 1259-1266s absorbed, K=6
    # flagged, laysa+laha missed). Here we anchor FORWARD: at every
    # inter-word gap >= max(0.5s, median) with same-ayah words on both
    # sides, test whether the post-gap audio re-sings the preceding phrase
    # by forced-aligning the SINGLE K-phrase (K=2..10, ayah-bounded) against
    # the post-gap window. Same bilateral/per-word/timing gates as the C
    # path (floor, floor-0.75, min_word_dur); skips cues already consumed by
    # stage 1. New repeat cues are spliced with is_repeat=True through the
    # same _spliced_cue path.
    import os as _os
    if _os.environ.get("QFA_DISABLE_FORWARD", "") == "":
        fixed = _forward_echo_pass(
            cues, combined_token_ids,
            blank_id, min_word_dur_frames, confidence_floor,
            median, accepted_fixes, fixed,
            log_probs_np, seconds_per_frame if seconds_per_frame is not None else 0.04,
            token_frame_spans, _spliced_cue,
            ext, path, fast_avg_logprob,
        )

    return fixed


def _forward_echo_pass(
    cues, combined_token_ids,
    blank_id, min_word_dur_frames, confidence_floor,
    median, accepted_fixes, fixed,
    log_probs_np, seconds_per_frame,
    token_frame_spans_fn, spliced_cue_fn,
    ext_main, path_main, fast_avg_logprob_fn,
):
    """Forward-anchored repeat-echo search (stage 2, see caller)."""
    from ..decode import fast_ctc_align_c
    from ..trellis import build_ext as _build_ext

    T = log_probs_np.shape[0]
    consumed_idx = set()
    for word_indices, _, _, _, _, _ in accepted_fixes.values():
        consumed_idx.update(word_indices)
    # cue positions may have shifted after splicing; work from the ORIGINAL
    # cues list (pre-splice) for anchor geometry, then splice into fixed.
    num_cues = len(cues)
    median_frames = float(median) if median and median > 0 else 1.0
    gap_thresh_frames = max(int(0.5 / seconds_per_frame), int(median_frames))

    def _positions_for(local_offset, count):
        return list(range(local_offset, local_offset + count))

    for i in range(num_cues - 1):
        gap = cues[i + 1]["start_frame"] - cues[i]["end_frame"]
        if gap < gap_thresh_frames:
            continue
        if cues[i].get("sura") != cues[i + 1].get("sura") or \
           cues[i].get("aya") != cues[i + 1].get("aya"):
            continue
        # K-word phrases ending at cue i (the pre-gap word), ayah-bounded.
        k_max = 1
        while (i - k_max + 1) >= 0 and \
                cues[i - k_max + 1].get("aya") == cues[i].get("aya") and \
                cues[i - k_max + 1].get("sura") == cues[i].get("sura"):
            k_max += 1
        k_max = max(1, k_max - 1)
        k_max = min(k_max, 10)
        if k_max < 2:
            continue
        # Post-gap audio window: from the ANCHOR cue's own end (not the
        # next cue's start) -- when the main pass absorbed a repeated
        # second pass, the echo audio lives INSIDE the "gap" between the
        # anchor end and the next cue start, and starting at next-cue
        # start skips over it entirely. Up to 3x the first-pass phrase
        # duration (generous: repeats can be slower).
        post_start = cues[i]["end_frame"] + 1
        if post_start >= T - 1:
            continue
        best = None  # (K, word_indices, s2, e2, tok_spans, ext1, path1, win_start)
        for K in range(2, k_max + 1):
            word_indices = list(range(i - K + 1, i + 1))
            if any(j in consumed_idx for j in word_indices):
                continue
            phrase_token_ids, _ = build_phrase_ids(
                word_indices, cues, combined_token_ids)
            if not phrase_token_ids:
                continue
            # First-pass expected duration from main-pass cue spans.
            first_dur = sum(cues[j]["end_frame"] - cues[j]["start_frame"] + 1
                            for j in word_indices)
            if first_dur <= 0:
                continue
            win_end = min(T - 1,
                          post_start + int(3.0 * first_dur) + 8)
            if win_end - post_start + 1 < len(phrase_token_ids):
                continue
            win_lp = np.ascontiguousarray(
                log_probs_np[post_start:win_end + 1], dtype=np.float32)
            ext1, path1, _ = fast_ctc_align_c(
                win_lp, phrase_token_ids, blank_id)
            if ext1 is None or path1 is None:
                continue
            n_states = len(ext1)
            f_seen = np.full(n_states, -1, dtype=np.int64)
            l_seen = np.full(n_states, -1, dtype=np.int64)
            for t, s in enumerate(path1):
                s = int(s)
                if 0 <= s < n_states:
                    if f_seen[s] < 0:
                        f_seen[s] = t
                    l_seen[s] = t
            # All phrase tokens must be visited.
            Lstan = len(phrase_token_ids)
            need = np.arange(Lstan) * 2 + 1
            if np.any(f_seen[need] < 0) or np.any(l_seen[need] < 0):
                continue
            s0 = int(f_seen[1])
            e0 = int(l_seen[2 * Lstan - 1])
            if e0 <= s0:
                continue
            # Onset gate: a genuine immediate repeat starts at the window
            # start (right after the gap). If the phrase's first token
            # first appears deep inside the window, the aligner fitted the
            # phrase to its own continuation downstream -- not a repeat.
            # 25 frames = 1.0s tolerance for gap decay/breath.
            if s0 > 25:
                continue
            # Strictly-after gate: the match must start AFTER the anchor
            # cue's own end. Otherwise the phrase matched its own
            # first-pass audio (self-match through overlapping/absorbed
            # cue spans), not a genuine second utterance.
            if post_start + s0 <= cues[i]["end_frame"] + 2:
                continue
            # Per-word spans + bilateral gates (mirror C thresholds).
            tok_spans = {}
            all_ok = True
            for m, j in enumerate(word_indices):
                nt = len(cues[j]["token_positions"])
                off = sum(len(cues[jj]["token_positions"])
                          for jj in word_indices[:m])
                ts, ss, se = token_frame_spans_fn(
                    _positions_for(off, nt), f_seen, l_seen)
                if ss is None or se is None or ts is None:
                    all_ok = False
                    break
                tok_spans[j] = (ss, se, ts)
            if not all_ok:
                continue
            avg = avg_logprob_along_path(win_lp, ext1, path1, s0, e0)
            # Forward stage demands a CLEAR margin above the floor (not just
            # clearing it): single-phrase forced alignment has no null
            # hypothesis, so a weak fit to post-gap audio would flag
            # everywhere. 1.0 nats bar + free-decode cross-check below.
            if avg < confidence_floor + 1.0:
                continue
            # Symmetry gate: the match must fit nearly as well as the
            # phrase's own first-pass fit. A wrong-words fit scores far
            # worse than the phrase sung normally; 1.5 nats separates
            # genuine second utterances from forced fits.
            _f0 = cues[word_indices[0]]["start_frame"]
            _f1 = cues[word_indices[-1]]["end_frame"]
            _first_avg = fast_avg_logprob_fn(_f0, _f1)
            dur = e0 - s0 + 1
            if __import__('os').environ.get('QFA_DEBUG_ANCHOR'):
                import sys as _sys
                print(f"[fwd] K={K} words={[cues[j]['word'] for j in word_indices]} "
                      f"bilat={avg:.3f} first={_first_avg:.3f} floor={confidence_floor:.3f} "
                      f"dur={dur} post_start={post_start}",
                      file=_sys.stderr, flush=True)
            if abs(avg - _first_avg) > 1.5:
                continue
            per_ok = True
            for j in word_indices:
                ss, se, _ = tok_spans[j]
                a = avg_logprob_along_path(win_lp, ext1, path1, ss, se)
                if a < confidence_floor - 0.75:
                    per_ok = False
                    break
            if not per_ok:
                continue
            if dur < len(word_indices) * int(min_word_dur_frames):
                continue

            # Free-decode cross-check (gate-5 philosophy): the window's
            # greedy collapse must look like the phrase twice. Kills fits
            # where forced alignment crams a phrase into unrelated audio.
            from ..decode import greedy_ctc_decode_ids as _greedy, token_id_levenshtein_ratio as _lev
            _dec = _greedy(win_lp, blank_id)
            _ph = list(phrase_token_ids)
            _db = _ph + _ph
            _rd = _lev(_dec, _db)
            _rs = _lev(_dec, _ph)
            if _rd < 0.40 or (_rd - _rs) < -0.43:
                continue
            # Best-K by score (bilateral + margin term), NOT widest:
            # widest-wins rewards long phrases absorbing non-repeat audio.
            _score = avg + (_rd - _rs) * 0.1
            if best is None or _score > best[0]:
                best = (_score, K, word_indices, s0, e0, tok_spans,
                        ext1, path1, post_start)
        if best is None:
            continue
        # Same-word coincidence veto: a second-pass word overlapping
        # (>50%) a main-pass cue of the SAME word text is the first pass
        # itself, not a repeat (kills chimeras whose tail re-matches
        # following first-pass words, e.g. wa-in..adl after a genuine
        # allah..shafee3 repeat).
        _bscore, _bK, _bwords, _bs0, _be0, _btok, _bext, _bpath, _bwin = best
        _veto = False
        for _j in _bwords:
            _w = cues[_j].get("word")
            _ms, _me = _btok[_j][0] + _bwin, _btok[_j][1] + _bwin
            for _q in range(num_cues):
                if cues[_q].get("word") != _w:
                    continue
                _qs, _qe = cues[_q]["start_frame"], cues[_q]["end_frame"]
                _ov = max(0, min(_me, _qe) - max(_ms, _qs) + 1)
                _spanlen = (_me - _ms + 1)
                if _spanlen > 0 and _ov / _spanlen > 0.5:
                    # Same word already timed here by the main pass: this
                    # "copy" is the first pass itself. Veto the WHOLE
                    # candidate (a chimera), not just the word.
                    _veto = True
                    break
            if _veto:
                break
        if _veto:
            continue
        _score, K, word_indices, s0, e0, tok_spans, ext1, path1, win_start = best
        if best is None:
            continue
        _, K, word_indices, s0, e0, tok_spans, ext1, path1, win_start = best
        # Splice: original cues keep is_repeat=False with re-cut spans is
        # NOT attempted (main-pass spans stand); append second-pass copies.
        per_word_copy2 = {}
        for j in word_indices:
            ss, se, ts = tok_spans[j]
            per_word_copy2[j] = (ss, se, ts)
        # Shift token spans into utterance frame coordinates.
        for j in word_indices:
            ss, se, ts = per_word_copy2[j]
            shifted = [(a + win_start, b + win_start) for (a, b) in ts]
            orig = cues[j]
            fixed.append(spliced_cue_fn(
                orig, ss + win_start, se + win_start, shifted,
                0, ext1, path1, True))
            fixed[-1]["repeat_stage"] = "forward-gap"
            consumed_idx.add(j)
    # Extension pass: for each STAGE-1 accepted fix, try widening RIGHT to
    # cover words PAST the anchor whose first pass was ABSORBED into the
    # stretched anchor span (e.g. Ghailan 6:99: K=3 wins on w7-9 at anchor
    # w9, but w10-11's first pass [48162,48217] sits absorbed inside the
    # same anchor span with their second pass [48280,48324] right after;
    # the doubled w7-11 fit over [48098,48324] is perfect: bilateral
    # -0.00, rd=1.0, margin +0.5). Re-test the widened phrase (accepted
    # words + extra right words) DOUBLED over the widened window; all
    # stage-1 gates apply. On pass, append the extra words' second-pass
    # copies. C-engine wins that legitimately END at the anchor are
    # unaffected (no rightward widening passes the doubled gates).
    import os as _os2
    _no_extend = _os2.environ.get("QFA_DISABLE_EXTEND", "") != ""
    for _anchor_i, (_w_idx, _a, _b, _ws, _e2, _p2) in accepted_fixes.items():
        if _no_extend:
            continue
        from .extend_fix import extend_fix_right
        extend_fix_right(
            _w_idx, cues, combined_token_ids, blank_id,
            min_word_dur_frames, confidence_floor,
            log_probs_np, seconds_per_frame, token_frame_spans_fn,
            spliced_cue_fn, fixed, consumed_idx, build_phrase_ids,
        )
    fixed.sort(key=lambda c: (c["start_frame"], c["end_frame"]))
    return fixed
