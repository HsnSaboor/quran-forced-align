"""Word-level cue extraction + repeat detection/local re-alignment.

Adapted from build_surah_srt.py's two-pass repeat-detection design, but
implemented as a forced-alignment re-run (against a DOUBLED local reference)
rather than a second banded-edit-distance pass.
"""
import numpy as np

from .constants import (
    DEFAULT_MAX_REPEAT_WINDOW_WORDS,
    FREE_DECODE_MIN_MARGIN,
    FREE_DECODE_MIN_RATIO_DOUBLED,
    GAP_ARTIFACT_MAX_FRAMES,
    GAP_ARTIFACT_MIN_MARGIN,
)
from .confidence import per_word_min_margin
from .decode import greedy_ctc_decode_ids, token_id_levenshtein_ratio
from .viterbi import avg_logprob_along_path, ctc_forced_align, frame_spans_from_path


def token_frame_spans(token_positions, first_seen, last_seen):
    """For a list of token positions (indices into a combined token-id
    list), return (per_token_spans, word_start, word_end) where
    per_token_spans[i] = (start_frame, end_frame) for token_positions[i]
    and word_start/word_end are the min/max over all of them. Token
    position p (an index into the combined token-id list) lives at
    extended-trellis label-state 2*p+1.

    Returns None if any token never got a state in the trellis (start<0),
    the same failure signal both `extract_word_frame_spans` and
    `_repeat_window_candidate` already relied on before this was factored
    out -- shared here so the phoneme-tier span data (per_token_spans) is
    computed identically, once, wherever a word's frame span is derived
    from token-level trellis states, instead of two near-duplicate min/max
    loops that could drift apart under future edits.
    """
    per_token_spans = []
    for p in token_positions:
        s = 2 * p + 1
        start, end = first_seen[s], last_seen[s]
        if start < 0 or end < 0:
            return None
        per_token_spans.append((int(start), int(end)))
    if not per_token_spans:
        return None
    word_start = min(s for s, _ in per_token_spans)
    word_end = max(e for _, e in per_token_spans)
    return per_token_spans, word_start, word_end


def extract_word_frame_spans(word_slots, first_seen, last_seen):
    """For each word slot with at least one token, compute (start_frame,
    end_frame) as the min/max over its tokens' extended-trellis label-state
    spans, plus the per-token spans themselves (`token_frame_spans`) --
    needed for the phoneme/letter-tier output (see `cells.py`), which
    would otherwise require re-deriving this from `first_seen`/`last_seen`
    a second time."""
    cues = []  # list of dicts: word, sura, aya, start_frame, end_frame, is_repeat, token_ids_global_pos
    for slot in word_slots:
        positions = slot["token_positions"]
        if not positions:
            continue
        spans = token_frame_spans(positions, first_seen, last_seen)
        if spans is None:
            continue
        per_token_spans, word_start, word_end = spans
        cues.append({
            "word": slot["word"],
            "sura": slot["sura"],
            "aya": slot["aya"],
            "is_ayah_final": slot["is_ayah_final"],
            "start_frame": word_start,
            "end_frame": word_end,
            "is_repeat": False,
            "token_positions": positions,
            "token_char_idx": slot["token_char_idx"],
            "letters": slot["letters"],
            "token_frame_spans": per_token_spans,
        })
    return cues


def _repeat_window_candidate(word_indices, cues, log_probs, combined_token_ids, blank_id,
                              window_start, window_end, min_word_dur_frames):
    """Try ONE candidate repeated-phrase hypothesis: the contiguous words at
    `word_indices` (indices into `cues`, in ascending order -- may be a
    single word or a multi-word phrase) doubled back-to-back and re-aligned
    against the frame span [window_start, window_end] of `log_probs`.

    This is the shared core the K=1 (single-word) doubling used to do
    inline; it's now extracted so the K-search loop in
    detect_and_fix_repeats can call it once per candidate window size
    without duplicating the trellis-construction/state-bookkeeping logic.

    Returns None if the doubled alignment fails outright, any word's tokens
    never got a state in the local trellis, or the two copies aren't
    timing-plausible (non-overlapping, each occupying >= min_word_dur_frames
    overall -- deliberately NOT scaled up by the number of words: this is
    the same single-word floor the original K=1 check used, applied to the
    whole phrase span, so it stays at least as permissive for K>1 as it was
    for K=1). Does NOT apply the acoustic-confidence gate -- that is the
    caller's job, so every K can be compared against the SAME floor on an
    equal footing.

    On success, returns a dict with the window's local ext/path/log_probs
    (for the caller to compute the confidence-gate averages), the whole-copy
    local spans (for the confidence gate), and a per-word breakdown of both
    copies' local (start, end) frame spans (for splicing individual word
    cues back in).
    """
    word_ntoks = [len(cues[j]["token_positions"]) for j in word_indices]
    if any(nt == 0 for nt in word_ntoks):
        return None
    offsets = []
    acc = 0
    for nt in word_ntoks:
        offsets.append(acc)
        acc += nt
    L = acc

    phrase_token_ids = []
    for j in word_indices:
        phrase_token_ids.extend(combined_token_ids[p] for p in cues[j]["token_positions"])
    doubled_ids = phrase_token_ids + phrase_token_ids

    window_log_probs = log_probs[window_start:window_end + 1]
    ext2, path2, margins2 = ctc_forced_align(window_log_probs, doubled_ids, blank_id)
    if ext2 is None:
        return None

    num_states = len(ext2)
    first_seen, last_seen = frame_spans_from_path(path2, num_states)

    def positions_for(local_offset, count):
        return list(range(local_offset, local_offset + count))

    copy1_positions = positions_for(0, L)
    copy2_positions = positions_for(L, L)
    copy1_spans = token_frame_spans(copy1_positions, first_seen, last_seen)
    copy2_spans = token_frame_spans(copy2_positions, first_seen, last_seen)
    if copy1_spans is None or copy2_spans is None:
        return None
    _copy1_token_spans, copy1_start_local, copy1_end_local = copy1_spans
    _copy2_token_spans, copy2_start_local, copy2_end_local = copy2_spans

    timing_plausible = (
        copy2_start_local > copy1_end_local
        and (copy1_end_local - copy1_start_local) >= min_word_dur_frames
        and (copy2_end_local - copy2_start_local) >= min_word_dur_frames
    )
    if not timing_plausible:
        return None

    per_word_copy1 = {}
    per_word_copy2 = {}
    for m, j in enumerate(word_indices):
        nt = word_ntoks[m]
        s1_spans = token_frame_spans(positions_for(offsets[m], nt), first_seen, last_seen)
        s2_spans = token_frame_spans(positions_for(L + offsets[m], nt), first_seen, last_seen)
        s1_token_spans, s1_start, s1_end = s1_spans
        s2_token_spans, s2_start, s2_end = s2_spans
        per_word_copy1[j] = (s1_start, s1_end, s1_token_spans)
        per_word_copy2[j] = (s2_start, s2_end, s2_token_spans)

    return {
        "window_log_probs": window_log_probs,
        "ext": ext2,
        "path": path2,
        "margins": margins2,
        "copy1_start_local": copy1_start_local,
        "copy1_end_local": copy1_end_local,
        "copy2_start_local": copy2_start_local,
        "copy2_end_local": copy2_end_local,
        "per_word_copy1": per_word_copy1,
        "per_word_copy2": per_word_copy2,
        # Exposed for the free-decode cross-check in detect_and_fix_repeats
        # (see FIX 5 in that function's docstring) -- built here already for
        # the doubled-reference forced alignment above, so the caller
        # doesn't need to rebuild them redundantly.
        "phrase_token_ids": phrase_token_ids,
        "doubled_ids": doubled_ids,
    }


def detect_and_fix_repeats(cues, log_probs, combined_token_ids, blank_id, ext, path,
                            low_ratio, high_ratio, min_word_dur_frames,
                            ayah_final_high_ratio_mult=1.5, confidence_margin=1.0,
                            max_repeat_window_words=DEFAULT_MAX_REPEAT_WINDOW_WORDS):
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

       We decode `cand["window_log_probs"]` (already sliced out for the
       forced-alignment call above -- no new audio decode or ONNX inference
       needed) with `decode.greedy_ctc_decode_ids` and compare the result,
       at token-id granularity, against the single-copy phrase
       (`phrase_token_ids`) and the doubled phrase (`doubled_ids`) using
       `decode.token_id_levenshtein_ratio`. A genuine two-occurrence window
       is required to look MORE like two copies than one, with real
       headroom, and to look reasonably like two copies on an absolute
       basis (ruling out a low-confidence decode that happens to edge out
       ratio_single by chance): reject unless
       `ratio_doubled >= FREE_DECODE_MIN_RATIO_DOUBLED` AND
       `ratio_doubled - ratio_single >= FREE_DECODE_MIN_MARGIN`.

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

    def high_cutoff(cue):
        return high_ratio * (ayah_final_high_ratio_mult if cue.get("is_ayah_final") else 1.0)

    def is_anomalous(cue):
        d = cue["end_frame"] - cue["start_frame"]
        return d < low_ratio * median or d > high_cutoff(cue) * median

    normal_avgs = [
        avg_logprob_along_path(log_probs, ext, path, c["start_frame"], c["end_frame"])
        for c in cues if not is_anomalous(c)
    ]
    baseline_avg = float(np.median(normal_avgs)) if normal_avgs else 0.0
    confidence_floor = baseline_avg - confidence_margin

    n = len(cues)
    consumed = set()           # word indices already spliced into an accepted fix
    accepted_fixes = {}        # trigger index i -> (word_indices, candidate_dict, window_start)
    margin = 2

    for i, c in enumerate(cues):
        if not is_anomalous(c) or not c["token_positions"]:
            continue

        # Natural structural bound on K: a real hifz-practice repeat never spans INTO a
        # different ayah (a reciter repeats a phrase or a whole ayah, not "half of ayah N
        # plus the start of ayah N+1"), so K can never usefully exceed the number of words
        # from the start of word i's own ayah up to and including i. This makes the search
        # correct for phrase repeats of ANY length (no arbitrary magic-number ceiling needed
        # for correctness) while still keeping the search cheap: it's bounded by the ayah's
        # own word count, and only runs at all on the rare words that already cleared the
        # anomaly-duration gate above. `max_repeat_window_words`, if not None, is an
        # additional hard cap purely for cost control in pathological cases -- see its
        # definition (DEFAULT_MAX_REPEAT_WINDOW_WORDS).
        words_left_in_aya = 1
        while i - words_left_in_aya >= 0 and cues[i - words_left_in_aya]["aya"] == c["aya"] and cues[i - words_left_in_aya]["sura"] == c["sura"]:
            words_left_in_aya += 1
        k_max = words_left_in_aya
        if max_repeat_window_words is not None:
            k_max = min(k_max, max_repeat_window_words)

        best = None  # (K, candidate_dict, bilateral_confidence, window_start)
        for K in range(1, k_max + 1):
            j0 = i - K + 1
            if j0 < 0:
                break  # larger K only decreases j0 further -- no point continuing
            if any(j in consumed for j in range(j0, i)):
                break  # window would overlap an earlier accepted fix; larger K only makes it worse

            # Local window: from the word just before the K-word candidate
            # phrase to the word just after the anomalous trigger word i, so
            # the re-alignment can only touch frames that aren't already
            # claimed by a neighbouring word's cue. K=1 (j0 == i) reduces to
            # exactly the original single-word window.
            #
            # At the surah/clip START boundary (j0 == 0, no previous cue),
            # fall back to a small margin before this word's own start.
            #
            # At the surah/clip END boundary (i == n-1, no next cue) we
            # canNOT fall back to a small margin past c["end_frame"]: that
            # main-pass estimate is exactly what's unreliable for an
            # ANOMALOUS word (that's the whole reason it's anomalous), and
            # in the "repeat with no pause" case the main pass settles into
            # trailing blank states early, understating end_frame by
            # several seconds -- confirmed empirically (ground-truth
            # test_C): the true second utterance's audio sat entirely past
            # end_frame + margin, so every K's search window silently
            # excluded the very audio it needed to find. Instead we extend
            # all the way to the true end of the available audio
            # (log_probs.shape[0] - 1) -- correct because at the true edge
            # of the clip there is no neighbouring cue's frames to avoid
            # touching, so there is no reason to hold back.
            window_start = cues[j0 - 1]["end_frame"] + 1 if j0 > 0 else max(0, cues[j0]["start_frame"] - margin)
            window_end = cues[i + 1]["start_frame"] - 1 if i < n - 1 else log_probs.shape[0] - 1
            window_start = max(0, window_start)
            window_end = min(log_probs.shape[0] - 1, window_end)
            if window_end <= window_start:
                continue

            word_indices = list(range(j0, i + 1))
            cand = _repeat_window_candidate(
                word_indices, cues, log_probs, combined_token_ids, blank_id,
                window_start, window_end, min_word_dur_frames,
            )
            if cand is None:
                continue

            # Fix 2's acoustic-confidence gate -- see docstring. Reject this
            # K unless BOTH copies clear the floor; either copy alone being
            # a weak mechanical artifact is disqualifying. Applied
            # identically for every K so widening the search never lowers
            # the bar, only the number of hypotheses tested against it.
            avg1 = avg_logprob_along_path(
                cand["window_log_probs"], cand["ext"], cand["path"],
                cand["copy1_start_local"], cand["copy1_end_local"],
            )
            avg2 = avg_logprob_along_path(
                cand["window_log_probs"], cand["ext"], cand["path"],
                cand["copy2_start_local"], cand["copy2_end_local"],
            )
            bilateral = min(avg1, avg2)
            if bilateral < confidence_floor:
                continue

            # Fix 4's gap-artifact secondary reject -- see docstring point 4.
            # A candidate that formally clears confidence_floor can STILL be
            # the mechanical CTC-trellis artifact Fix 2 was written to catch
            # (docstring point 2's "almost-exactly-constant ~0.04s gap...
            # exactly the CTC trellis's mandatory blank-frame separator, not
            # a genuine acoustic pause") if it only barely cleared the floor.
            # Reject only when BOTH signals are present -- a minimal gap
            # AND a marginal confidence margin -- so a 1-frame gap with a
            # comfortably large margin (strong acoustic evidence) is not
            # blanket-banned, and a wide, plausible gap is never touched by
            # this check regardless of margin.
            gap_frames = cand["copy2_start_local"] - cand["copy1_end_local"]
            margin_above_floor = bilateral - confidence_floor
            if gap_frames <= GAP_ARTIFACT_MAX_FRAMES and margin_above_floor < GAP_ARTIFACT_MIN_MARGIN:
                continue

            # Fix 5's free-decode cross-check -- see docstring point 5. An
            # UNCONSTRAINED greedy decode of this same window (no reference
            # bias) must itself look more like TWO copies of the phrase
            # than ONE, with real headroom, and must look reasonably like
            # two copies on an absolute basis. Computed on
            # `cand["window_log_probs"]`, which is already sliced out above
            # -- no new inference needed.
            decoded_ids = greedy_ctc_decode_ids(cand["window_log_probs"], blank_id)
            ratio_single = token_id_levenshtein_ratio(decoded_ids, cand["phrase_token_ids"])
            ratio_doubled = token_id_levenshtein_ratio(decoded_ids, cand["doubled_ids"])
            free_decode_passes = (
                ratio_doubled >= FREE_DECODE_MIN_RATIO_DOUBLED
                and (ratio_doubled - ratio_single) >= FREE_DECODE_MIN_MARGIN
            )
            if not free_decode_passes:
                continue

            # Tie-break rule (see docstring point 3): keep the K with the
            # highest bilateral confidence; smaller K wins exact ties since
            # we iterate K ascending and only replace `best` on a strict
            # improvement.
            if best is None or bilateral > best[2]:
                best = (K, cand, bilateral, window_start)

        if best is None:
            continue

        K, cand, _bilateral, window_start = best
        word_indices = list(range(i - K + 1, i + 1))
        accepted_fixes[i] = (word_indices, cand, window_start)
        consumed.update(word_indices)

    if not accepted_fixes:
        return cues

    def _shift_token_spans(token_spans, offset):
        return [(s + offset, e + offset) for s, e in token_spans]

    def _spliced_cue(orig, start_local, end_local, token_spans_local, window_start, cand, is_repeat):
        # Confidence signals for a repeat-spliced word MUST be computed
        # against `cand`'s own LOCAL doubled-reference re-alignment
        # (window_log_probs/ext/path/margins), never the main pass's --
        # this word's frames live in a different, locally re-derived
        # trellis than the one the main-pass ext/path/margins describe, so
        # reusing the main pass's arrays here would silently score the
        # wrong symbols. Pre-computing these here (rather than leaving them
        # for confidence.flag_low_confidence_words to fill in generically)
        # is what lets that function treat "already has avg_logprob" as
        # the signal to skip a cue -- see its docstring.
        return {
            **orig,
            "start_frame": start_local + window_start,
            "end_frame": end_local + window_start,
            "is_repeat": is_repeat,
            "token_frame_spans": _shift_token_spans(token_spans_local, window_start),
            "avg_logprob": avg_logprob_along_path(
                cand["window_log_probs"], cand["ext"], cand["path"], start_local, end_local,
            ),
            "min_decision_margin": per_word_min_margin(cand["margins"], start_local, end_local),
        }

    fixed = [c for j, c in enumerate(cues) if j not in consumed]
    for word_indices, cand, window_start in accepted_fixes.values():
        for j in word_indices:
            s1, e1, tok_spans1 = cand["per_word_copy1"][j]
            s2, e2, tok_spans2 = cand["per_word_copy2"][j]
            orig = cues[j]
            fixed.append(_spliced_cue(orig, s1, e1, tok_spans1, window_start, cand, is_repeat=False))
            fixed.append(_spliced_cue(orig, s2, e2, tok_spans2, window_start, cand, is_repeat=True))

    return fixed
