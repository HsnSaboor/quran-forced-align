"""Per-ayah/word Uthmani-text <-> phoneme mapping logic.

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
import quran_transcript as qt

from ..constants import MOSHAF


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
