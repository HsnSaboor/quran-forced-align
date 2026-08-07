"""Per-ayah word<->phoneme reference builder + whole-surah combined
reference.

Fixes a real bug found in build_surah_srt.py's build_text_reference() (that
older, superseded script is not modified/used here at all -- fixed
independently in this package instead, see the docstring below).

NOTE ON NAMING: the original monolithic script (forced_align_srt.py) named
these functions with a `_correct` suffix (`build_text_reference_correct`,
etc.) to disambiguate them from the buggy sibling of the same name living in
the older build_surah_srt.py script. That buggy sibling does not exist in
this package, so the suffix is meaningless clutter here and has been
dropped.
"""
import logging

import quran_transcript as qt

from .constants import MOSHAF
from .tokenizer import tokenize_with_char_starts

logger = logging.getLogger(__name__)


def _rule_names(tajweed_rules):
    """Serialize a quran_transcript TajweedRule list to plain strings (the
    rule class name, e.g. "Ghonnah", "MaddRule") for stable, dependency-free
    JSON output -- callers of this package's JSON output shouldn't need the
    quran_transcript package installed just to interpret a tajweed tag."""
    if not tajweed_rules:
        return []
    return [type(r).__name__ for r in tajweed_rules]


def build_text_reference(text, words, sura_idx, aya_idx):
    """Given known Uthmani `text` and its pre-split `words`, return the
    phoneme<->word mapping, indexed correctly, plus per-character tajweed
    and silent-letter data needed for the letter-tier output (see
    `char_info` below).

    BUG FOUND IN build_surah_srt.py's build_text_reference() (not modified,
    per task instructions -- fixed here instead): that function calls
    `char_to_word(m.pos[0])` for each mapping `m` returned by
    `qt.quran_phonetizer`, but `m.pos` is a span into the OUTPUT
    phonemes-with-spaces string, not into the input `text` string that
    `char_to_word` (built from `word_spans`, which are spans into `text`)
    expects. These two coordinate spaces only coincide by accident for
    words with no shadda-doubling/zero-width mappings, which is why it
    wasn't caught earlier by informal testing: confirmed by direct
    inspection this silently maps ~9% of words to `phoneme_to_word=None`
    even in Al-Fatiha (3/34 words: 'ٱلرَّحِيمِ' in 1:1, 'وَلَا' and
    'ٱلضَّآلِّينَ' in 1:7), and far more in surah 67 (26/30 ayahs have a
    text/phonemes-word-count mismatch under the buggy indexing). A `None`
    word attribution means that word's phoneme tokens silently get no
    timing -- exactly the kind of silent data-quality bug forced alignment
    is supposed to eliminate, so it has to be fixed here rather than
    inherited.

    Fix: `qt.quran_phonetizer`'s `mappings` list has exactly one entry per
    character of the ORIGINAL `text` (confirmed empirically: len(mappings)
    == len(text) for every ayah of Al-Fatiha and surah 67 checked), and
    each mapping's `.pos` gives that char's corresponding span in the
    OUTPUT phonemes-with-spaces string (a span of length 0, 1, or more
    output chars). So: build `text_char_to_word` by running the (correct)
    char_to_word() over `range(len(text))` directly; separately build
    `output_pos_to_text_char` by scattering each mapping index across its
    own `.pos` span in the output string; then for every non-space
    character of the output string, look up which text char produced it
    and from there which word it belongs to. This reproduces exactly what
    build_surah_srt.py's build_text_reference() was trying to do, just
    with the coordinate spaces matched up correctly.

    `mappings[i].deleted` (True for a silently-dropped Uthmani character,
    e.g. hamzat al-wasl) and `mappings[i].tajweed_rules` (madd/ghunnah/
    qalqalah/idgham rules quran_transcript already attaches per character)
    are exposed here as `char_info` -- one entry per character of `text`,
    each `{"char", "deleted", "tajweed_rules"}` -- so the letter-tier
    output (see `cells.build_letter_tier`) doesn't need to re-derive this
    from raw mappings itself.
    """
    out = qt.quran_phonetizer(text, MOSHAF)
    mappings = out.mappings
    if len(mappings) != len(text):
        raise ValueError(
            f"quran_phonetizer returned {len(mappings)} mappings for a "
            f"{len(text)}-char text -- expected 1:1; cannot safely attribute "
            f"phonemes to words for {text!r}"
        )

    word_spans = []
    cursor = 0
    for w in words:
        start = text.index(w, cursor)
        end = start + len(w)
        word_spans.append((start, end))
        cursor = end

    # word_spans are sorted, non-overlapping ranges covering `text` in
    # order (built by the loop just above via sequential text.index()
    # calls), so a single forward cursor walk assigns every char to its
    # word in O(len(text) + len(words)) total -- a linear re-scan of
    # word_spans per character (the original approach) was
    # O(len(text) * len(words)), needlessly quadratic for no benefit since
    # nothing here requires random-access lookup.
    text_char_to_word = [None] * len(text)
    wi = 0
    for charpos in range(len(text)):
        while wi < len(word_spans) and charpos >= word_spans[wi][1]:
            wi += 1
        if wi < len(word_spans) and word_spans[wi][0] <= charpos:
            text_char_to_word[charpos] = wi

    spaced_phonemes = out.phonemes
    output_pos_to_text_char = [None] * len(spaced_phonemes)
    for text_idx, m in enumerate(mappings):
        s, e = m.pos
        for p in range(s, e):
            output_pos_to_text_char[p] = text_idx

    phonemes_stripped_chars = []
    phoneme_to_word = []
    phoneme_to_char = []
    for p, ch in enumerate(spaced_phonemes):
        if ch == " ":
            continue
        text_idx = output_pos_to_text_char[p]
        # Every non-space character of the phonemizer's output is expected
        # to be covered by exactly one mapping's `.pos` span (this is the
        # load-bearing assumption the whole word/char-attribution scheme
        # above depends on -- `len(mappings) == len(text)` alone only
        # confirms the COUNT matches, not that the spans actually TILE the
        # output string with no gaps). A gap here would silently drop this
        # phoneme token from every downstream list (phoneme_to_word,
        # phoneme_to_char, and therefore token_positions in
        # build_combined_reference) with no error -- exactly the kind of
        # silent data-quality bug this module's docstring says forced
        # alignment is supposed to eliminate, so fail loudly instead.
        if text_idx is None:
            raise ValueError(
                f"quran_phonetizer output character {ch!r} at output position {p} in "
                f"{spaced_phonemes!r} is not covered by any mapping's .pos span -- "
                f"cannot safely attribute this phoneme to a source character for {text!r}"
            )
        wi = text_char_to_word[text_idx]
        phonemes_stripped_chars.append(ch)
        phoneme_to_word.append(wi)
        phoneme_to_char.append(text_idx)

    char_info = [
        {
            "char": text[i],
            "deleted": m.deleted,
            "tajweed_rules": _rule_names(m.tajweed_rules),
            # Populated later, only for the first/last letter of a word, by
            # build_combined_reference's cross-ayah-boundary pass -- present
            # here with an empty default so every letter dict has the same
            # keys regardless of whether a boundary rule ends up applying.
            "boundary_tajweed_rules": [],
        }
        for i, m in enumerate(mappings)
    ]

    return {
        "words": words,
        "word_spans": word_spans,
        "phonemes": "".join(phonemes_stripped_chars),
        "phoneme_to_word": phoneme_to_word,
        "phoneme_to_char": phoneme_to_char,
        "char_info": char_info,
        "sura_idx": sura_idx,
        "aya_idx": aya_idx,
    }


def build_ayah_reference(sura_idx, aya_idx):
    aya = qt.Aya(sura_idx, aya_idx)
    d = aya.get()
    return build_text_reference(d.uthmani, d.uthmani_words, sura_idx, aya_idx)


def build_istiaatha_reference(sura_idx):
    aya = qt.Aya(sura_idx, 1)
    d = aya.get()
    text = getattr(d, "istiaatha_uthmani", None)
    if not text:
        return None
    words = text.split()
    return build_text_reference(text, words, sura_idx, 0)


def build_surah_reference(sura_idx, include_istiaatha=True):
    aya = qt.Aya(sura_idx, 1)
    d = aya.get()
    n_ayat = d.num_ayat_in_sura
    refs = []
    if include_istiaatha:
        preamble = build_istiaatha_reference(sura_idx)
        if preamble:
            refs.append(preamble)
    for aya_idx in range(1, n_ayat + 1):
        refs.append(build_ayah_reference(sura_idx, aya_idx))
    return refs


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
    except ValueError:
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
        #
        # Logged (not silently swallowed) precisely because this except
        # clause catches a bare ValueError -- the SAME exception type
        # build_text_reference's OWN internal invariant-violation checks
        # raise (see its "not 1:1" and "no mapping covers this output
        # character" checks above) for reasons that have NOTHING to do
        # with muqattaat. Without a log line here, a future regression in
        # either this package or quran_transcript that raises ValueError
        # for a genuinely different reason would be silently misattributed
        # to "known muqattaat limitation" and produce no signal that
        # something is actually broken.
        logger.warning(
            "skipping cross-ayah-boundary tajweed probe for %r|%r: quran_phonetizer "
            "raised ValueError when phonetizing one of these words standalone "
            "(known limitation for muqattaat-letter words; if this word is NOT "
            "muqattaat letters, this may indicate a real bug)",
            prev_word, next_word,
        )
        return [], []

    next_word_offset = len(prev_word) + 1  # +1 for the joining space

    def _diff(joint_span, isolated_span):
        diffs = []
        for char_idx, (joint_char, isolated_char) in enumerate(zip(joint_span, isolated_span)):
            joint_rules = set(joint_char["tajweed_rules"])
            isolated_rules = set(isolated_char["tajweed_rules"])
            if joint_rules != isolated_rules:
                diffs.append((char_idx, sorted(joint_rules - isolated_rules)))
        return diffs

    prev_extra = _diff(joint["char_info"][:len(prev_word)], prev_isolated["char_info"])
    next_extra = _diff(joint["char_info"][next_word_offset:next_word_offset + len(next_word)],
                        next_isolated["char_info"])
    return prev_extra, next_extra


def build_combined_reference(sura_idx, tok2id, max_token_len, include_boundary_tajweed=True):
    """Concatenate the whole surah's (istiaatha + all ayahs) reference
    phoneme sequences into ONE token-id list, and build a parallel list of
    "word slots" (one per real Arabic word across the whole surah) each
    carrying the list of global positions (indices into the combined
    token-id list) of the tokens that belong to that word, plus per-token
    character attribution and per-character tajweed/silent-letter data
    needed for the letter/phoneme-tier output (see `cells.py`).

    This is what makes the single whole-surah-at-once Viterbi pass possible:
    the reference position space is flat and global, so the alignment
    doesn't need to know about ayah boundaries at all -- they fall out
    naturally from which word slot each token belongs to.
    """
    refs = build_surah_reference(sura_idx)
    combined_token_ids = []
    word_slots = []  # each: {"word","sura","aya","is_ayah_final","token_positions",
                      #        "token_char_idx","letters","boundary_tajweed_rules"}

    for ref in refs:
        ids, char_starts = tokenize_with_char_starts(ref["phonemes"], tok2id, max_token_len)
        word_slot_base = len(word_slots)
        n_words_in_ref = len(ref["words"])
        for wi, (word_start, word_end) in enumerate(ref["word_spans"]):
            word_slots.append({
                "word": ref["words"][wi],
                "sura": ref["sura_idx"],
                "aya": ref["aya_idx"],
                # Last word of this ayah (or of the istiaatha preamble)'s
                # reference -- flagged so the repeat-anomaly detector can
                # give it a pass on natural waqf (pause) lengthening rather
                # than treating it identically to a mid-ayah word (see
                # detect_and_fix_repeats).
                "is_ayah_final": wi == n_words_in_ref - 1,
                "token_positions": [],
                # Word-local character index (0-based within this word's
                # own text span) each token_positions entry maps to --
                # parallel array, needed to group phoneme-tier tokens into
                # letter-tier spans downstream (see cells.build_letter_tier).
                "token_char_idx": [],
                # Per-character skeleton for this word: {"char","deleted",
                # "tajweed_rules"} -- independent of alignment timing, so
                # this is fully known here already (deleted/silent chars
                # never get any token_positions entry, since they produce
                # zero phoneme output).
                "letters": ref["char_info"][word_start:word_end],
            })
        for local_tok_idx, cs in enumerate(char_starts):
            wi = ref["phoneme_to_word"][cs] if cs < len(ref["phoneme_to_word"]) else None
            char_idx = ref["phoneme_to_char"][cs] if cs < len(ref["phoneme_to_char"]) else None
            global_pos = len(combined_token_ids) + local_tok_idx
            if wi is not None:
                # char_idx is always non-None here: `wi` (looked up from
                # `phoneme_to_word[cs]`, built in lockstep with
                # `phoneme_to_char[cs]` in build_text_reference's loop
                # above) can only be non-None when `text_char_to_word` /
                # `phoneme_to_char` were themselves non-None for this same
                # position -- so no `is not None` guard is needed on
                # char_idx specifically.
                slot = word_slots[word_slot_base + wi]
                slot["token_positions"].append(global_pos)
                word_start = ref["word_spans"][wi][0]
                slot["token_char_idx"].append(char_idx - word_start)
        combined_token_ids.extend(ids)

    # Attach wasl-only cross-boundary tajweed rules directly onto whichever
    # specific letter(s) they apply to (not always the boundary-adjacent
    # character -- see _boundary_bridge_rules) rather than a single
    # per-word field, since a word sits on the "next" side of one boundary
    # and the "prev" side of the next boundary simultaneously -- a
    # per-word field would let the second assignment silently clobber the
    # first.
    #
    # Only probed at REAL ayah boundaries (is_ayah_final words), not every
    # adjacent word pair: words within the same ayah were already
    # phonetized TOGETHER by the single per-ayah quran_phonetizer call in
    # the loop above, so any cross-word tajweed effect between them is
    # already correctly captured there -- re-probing every intra-ayah pair
    # here would be pure wasted work (and, at whole-surah scale, a lot of
    # it: ~6,100 redundant calls instead of ~286 for Al-Baqarah).
    # Costs one extra quran_phonetizer call per ayah boundary (~20-100ms
    # each depending on word length) on top of the per-ayah phonetization
    # already paid above -- for Al-Baqarah (~286 boundaries) this roughly
    # doubles reference-building time (~50s -> ~90s), which is still
    # negligible next to that surah's ONNX inference + Viterbi cost (the
    # actual dominant cost of the pipeline), but `include_boundary_tajweed`
    # is exposed so callers who don't need this tier (e.g. a fast
    # smoke-test run) can skip it.
    if include_boundary_tajweed:
        for i in range(len(word_slots) - 1):
            if not word_slots[i]["is_ayah_final"]:
                continue
            prev_extra, next_extra = _boundary_bridge_rules(word_slots[i]["word"], word_slots[i + 1]["word"])
            for char_idx, rules in prev_extra:
                word_slots[i]["letters"][char_idx]["boundary_tajweed_rules"] = rules
            for char_idx, rules in next_extra:
                word_slots[i + 1]["letters"][char_idx]["boundary_tajweed_rules"] = rules

    return combined_token_ids, word_slots
