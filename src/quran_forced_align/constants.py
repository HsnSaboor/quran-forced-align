"""Shared constants and calibrated thresholds for the forced-alignment
pipeline. Ported verbatim (including calibration-rationale comments) from
the original monolithic `forced_align_srt.py` -- these comments encode real
empirical findings (exact nats margins, frame counts) from independent
verification runs and must not be paraphrased away.
"""
from quran_transcript.phonetics.moshaf_attributes import MoshafAttributes

SAMPLE_RATE = 16000

MOSHAF = MoshafAttributes(
    rewaya="hafs",
    madd_monfasel_len=4,
    madd_mottasel_len=4,
    madd_mottasel_waqf=4,
    madd_aared_len=4,
)

FRAME_SHIFT_SEC = 0.01  # 10ms fbank hop -- matches build_surah_srt / model export convention
MIN_WORD_DUR = 0.15     # floor for degenerate (near-zero-length) word cues, matches build_surah_srt

# Optional HARD SAFETY CAP (in words) on the repeated-phrase K-search in detect_and_fix_repeats,
# on top of the natural (ayah-boundary) bound described there. None means "no extra cap" -- the
# search is bounded by how many words remain in the current ayah, which is the correct
# structural bound for a real hifz-practice repeat (a reciter repeats a phrase or a whole ayah,
# never spanning INTO a different ayah) and needs no arbitrary magic number to be correct for
# "any amount of repeated words." This exists purely as a cost-control escape valve for
# pathologically long ayahs if the search ever needs to be capped in practice; it is not needed
# for correctness and defaults to off.
DEFAULT_MAX_REPEAT_WINDOW_WORDS = None

# GAP-ARTIFACT secondary reject thresholds for the acoustic-confidence gate in
# detect_and_fix_repeats (see that function's docstring point 2 for the full
# background, and point 4 for this specific tightening). Calibrated against
# an independent verification pass on surah 67 that found the confidence
# gate ALONE still let one mechanical-artifact split through: ayah 8's K=2
# split (words 'ٱلْغَيْظِ'+'كُلَّمَآ') cleared confidence_floor by only 0.034
# nats AND had a gap_frames=1 (~0.04s) separation between copy1_end and
# copy2_start -- exactly the "almost-exactly-constant ~0.04s gap (one output
# frame)" CTC-trellis-blank-separator signature the docstring already
# documents, here showing up despite formally clearing the floor. Every
# CONFIRMED genuine catch in the same investigation (ground-truth test_B/
# test_C's 4-word recoveries, this file's own ayah-28 K=3 widening) cleared
# the floor by >=0.91 nats and had a gap of >=7 frames (>=0.28s) -- so
# neither threshold below touches them (their gaps alone already exceed
# GAP_ARTIFACT_MAX_FRAMES, which is checked first).
GAP_ARTIFACT_MAX_FRAMES = 1  # 1 output frame = seconds_per_output_frame (~0.04s at this
# model's 4x-subsampled decode_chunk_len, see run_streaming_log_probs) -- the
# smallest gap the trellis can produce at all between two adjacent copies
# (timing_plausible in _repeat_window_candidate already requires
# copy2_start_local > copy1_end_local strictly, so 1 frame is the floor, not
# an arbitrary cutoff).
GAP_ARTIFACT_MIN_MARGIN = 0.5  # nats above confidence_floor. Must be > the 0.034
# that slipped through (so the ayah-8 case above is actually rejected) and
# comfortably below the >=0.91 every genuine catch cleared (so this isn't
# calibrated to the single failing data point with zero headroom) -- 0.5
# sits roughly midway on a log scale between the two, rejecting both the
# 0.034 case and a second same-signature case found in the same
# investigation (idx=202, 'صَـٰٓفَّـٰتٍۢ', K=1, margin 0.42) without coming
# anywhere near the genuine catches' margins.

# FREE-DECODE CROSS-CHECK thresholds for detect_and_fix_repeats (see that
# function's docstring point 5 for the full background). Calibrated against
# an UNCONSTRAINED greedy CTC decode (no reference bias -- decode.py's
# greedy_ctc_decode_ids) of the ACTUAL K-search windows detect_and_fix_repeats
# itself constructs (not an ad-hoc padded clip -- an earlier standalone
# verification script that used a narrow +/-1.5s pad around only the flagged
# SECOND-copy window structurally could not see an earlier first occurrence
# several seconds before it, and wrongly suggested all 3 surah cases below
# were false positives; re-running the check on the real, wider window this
# module actually searches changed the verdict for 2 of the 3 -- see the
# table below. Always calibrate against the code path that will actually
# run, not a hand-rolled approximation of it).
#
# Per-K ratio_single / ratio_doubled from the real K-search loop (best-K
# shown for the multi-K cases; every K tried is documented in this session's
# notes, not reproduced here):
#
#   case                                    K   ratio_single  ratio_doubled  margin  verdict
#   surah 67 aya 28 (فَمَن يُجِيرُ ٱلْكَـٰفِرِينَ)  3     0.5000        0.9615    +0.4615  GENUINE -- free decode independently
#                                                                                          shows the phrase twice, continuing
#                                                                                          into the ayah's real next words only
#                                                                                          after the second copy. NOT a false
#                                                                                          positive; correctly still flagged.
#   surah 68 aya 26 (إِنَّا لَضَآلُّونَ)            2     0.8889        0.4444    -0.4444  FALSE POSITIVE -- free decode shows
#                                                                                          the phrase only ONCE; correctly
#                                                                                          rejected by this gate (flags dropped
#                                                                                          from 2 to 0 for this ayah).
#   surah 71 aya 7  (جَعَلُوٓا۟ أَصَـٰبِعَهُمْ فِى ءَاذَانِهِمْ) 4  0.5000  1.0000  +0.5000  GENUINE -- same as surah 67 aya 28,
#                                                                                          the free decode independently
#                                                                                          confirms two occurrences.
#   test_B (GENUINE 4-word repeat, pause)          4     0.4615        0.9231    +0.4615  GENUINE (ground truth, ratified)
#   test_C (GENUINE 4-word repeat, no pause)       4     0.4615        0.9231    +0.4615  GENUINE (ground truth, ratified)
#
# So of the 3 surah cases originally suspected as false positives, only
# surah 68 aya 26 actually is one under this independent cross-check; surah
# 67 aya 28 and surah 71 aya 7 are confirmed genuine by the SAME check and
# must not be rejected. The one confirmed false positive has a NEGATIVE
# margin and ratio_doubled well below 0.5; every genuine case (both
# confirmed-genuine surah ayahs and both ground-truth fixtures) clears
# ratio_doubled >= 0.92 with a margin >= 0.46 -- a clean separation with
# real headroom on both sides of this 5-point calibration set.
FREE_DECODE_MIN_RATIO_DOUBLED = 0.75  # sits below every genuine case's ratio_doubled
# (>=0.9231) and above the one confirmed false positive's ratio_doubled (0.4444).
FREE_DECODE_MIN_MARGIN = 0.15  # (ratio_doubled - ratio_single) must clear this.
# The confirmed false positive has a NEGATIVE margin (-0.4444); every genuine
# case clears +0.46. 0.15 sits comfortably above 0 (so it never lets a
# negative-margin case through) and well below the genuine cases' margins,
# with no need to ride the zero line.
