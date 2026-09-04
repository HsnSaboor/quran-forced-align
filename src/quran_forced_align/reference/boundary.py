import logging

from .text import build_text_reference

logger = logging.getLogger(__name__)


_isolated_word_reference_cache = {}


def _isolated_word_reference(word):
    """Memoized `build_text_reference(word, [word], 0, 0)` -- every word
    (except the very first/last of the surah) is the "next" word of one
    ayah boundary and the "prev" word of the following one, so
    `_boundary_bridge_rules` would otherwise re-phonetize the SAME word in
    isolation twice with identical inputs/outputs. Caching halves that
    redundant half of the boundary-bridging cost (the joint two-word calls
    are NOT cached, since each is genuinely unique to its specific
    boundary pair). Module-level cache (not cleared between surahs) is
    safe and correct here: `build_text_reference`'s output for a given
    word text + the fixed MOSHAF config is a pure function of that text,
    with no surah-specific state -- the same word text phonetizes
    identically wherever it occurs.
    """
    cached = _isolated_word_reference_cache.get(word)
    if cached is None:
        cached = build_text_reference(word, [word], 0, 0)
        _isolated_word_reference_cache[word] = cached
    return cached


def _boundary_bridge_rules(prev_word, next_word):
    """Cross-boundary tajweed rules that only surface when the last word of
    one ayah (or the istiaatha preamble) and the first word of the next are
    phonetized TOGETHER, vs. each in isolation as `build_surah_reference`
    already does per-ayah.

    WHY THIS EXISTS: quran_transcript's madd rules depend on whether the
    word is followed by more speech (continuous/wasl recitation) or treated
    as the end of an utterance (paused/waqf recitation) -- confirmed
    empirically: phonetizing "يَخَافُ" alone tags its madd letter
    `AaredMaddRule` (the longer، pause-lengthened "aared" madd), while
    phonetizing "يَخَافُ ٱللَّهَ" together tags the SAME letter
    `NormalMaddRule` (the shorter madd used when speech continues past it)
    -- exactly the wasl-vs-waqf madd distinction real tajweed requires.
    Per-ayah phonetization (what `build_surah_reference` already does)
    always treats each ayah's last word as if a pause follows it, which is
    only sometimes true for real continuous recitation across an ayah
    boundary. (NOTE: quran_transcript's ghunnah/idgham/ikhfaa handling
    already looks across the word-separating space at the PHONEME-TEXT
    level regardless of this function -- e.g. "مَنْ يَقُولُ" phonetized as
    one string already merges the noon into the following yaa -- but that
    package does not currently attach a `TajweedRule` tag for those
    substitutions the way it does for madd/qalqalah, so there is nothing
    for this function to recover for that rule family; only madd/qalqalah
    differences are tag-observable this way today.)

    Re-running the FULL surah (or even just consecutive whole ayahs)
    through one phonetizer call to catch this is NOT viable:
    `quran_phonetizer` scales super-linearly with input length (empirically:
    20 concatenated ayahs of Al-Baqarah took ~13s vs ~0.3s/ayah phonetized
    separately -- a ~40x slowdown for that chunk alone, and a whole-surah
    call for Al-Baqarah did not finish in over 2 minutes). Restricting the
    joint call to just the two boundary WORDS (not the whole ayahs) keeps
    this bounded and cheap regardless of ayah length -- empirically ~20ms
    per boundary, negligible next to the ~0.3s/ayah already being paid for
    the isolated per-ayah calls.

    Returns (prev_extra, next_extra): each a list of (char_idx, rule_names)
    pairs -- char_idx is a word-local index into `prev_word`/`next_word`
    (not just the boundary-adjacent character: a madd rule attaches to
    the madd LETTER, which may sit one or more positions before the word's
    literal last character, e.g. a trailing diacritic) -- covering every
    character whose tajweed_rules differ between the JOINT (this word +
    its neighbour) and ISOLATED (this word alone, matching what the
    per-ayah reference already computed) phonetization. Returned
    separately from each word's normal (isolated, waqf-consistent)
    `char_info` tajweed rules, rather than merged into them, because
    whether a real reciter actually paused at this specific ayah boundary
    is NOT decidable from text alone (real recitations vary): a consumer
    that wants the wasl-vs-waqf choice resolved against real audio (e.g.
    by measuring silence at this boundary) can combine this with
    `char_info`'s waqf-consistent base rules as needed; one that only
    wants the safe waqf-consistent reading can ignore this field entirely.
    """
    try:
        joint = build_text_reference(f"{prev_word} {next_word}", [prev_word, next_word], 0, 0)
        prev_isolated = _isolated_word_reference(prev_word)
        next_isolated = _isolated_word_reference(next_word)
    except (ValueError, IndexError, Exception):
        # A handful of Quranic words (e.g. the disjointed "muqattaat"
        # letters that open some surahs, like "الٓمٓ") are phonetized with
        # special-cased multi-letter-name expansions that quran_transcript
        # only applies correctly in certain contexts, breaking
        # build_text_reference's 1:1 mapping assumption when such a word is
        # phonetized STANDALONE (outside its normal ayah context) --
        # confirmed on surah 2's "الٓمٓ" opening. This is a real, narrow
        # limitation of the underlying phonemizer package's
        # standalone-word handling, not a bug in the boundary-bridging
        # logic itself: there is no correct answer to compute here without
        # also reproducing that word's full ayah context, which would
        # defeat the point of a cheap word-pair probe. Skip this boundary
        # rather than crashing the whole surah's alignment over what is,
        # structurally, an ADDITIONAL tajweed-tagging enhancement on top
        # of an already-correct base phonemization (the per-ayah
        # `char_info` this boundary check would have augmented is
        # entirely unaffected).
        logger.warning(
            "skipping cross-ayah-boundary tajweed probe for %r|%r: quran_phonetizer "
            "raised an exception when phonetizing one of these words standalone "
            "(known limitation for muqattaat-letter words; if this word is NOT "
            "muqattaat letters, this may indicate a real bug)",
            prev_word, next_word,
        )
        return [], []

    next_word_offset = len(prev_word) + 1  # +1 for the joining space

    def _diff(joint_span, isolated_span):
        diffs = []
        for char_idx, (joint_char, isolated_char) in enumerate(zip(joint_span, isolated_span)):
            if joint_char["tajweed_rules"] == isolated_char["tajweed_rules"]:
                continue
            joint_rules = set(joint_char["tajweed_rules"])
            isolated_rules = set(isolated_char["tajweed_rules"])
            if joint_rules != isolated_rules:
                diffs.append((char_idx, sorted(joint_rules - isolated_rules)))
        return diffs

    prev_extra = _diff(joint["char_info"][:len(prev_word)], prev_isolated["char_info"])
    next_extra = _diff(joint["char_info"][next_word_offset:next_word_offset + len(next_word)],
                        next_isolated["char_info"])
    return prev_extra, next_extra
