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
import quran_transcript as qt

from .constants import MOSHAF


def build_text_reference(text, words, sura_idx, aya_idx):
    """Given known Uthmani `text` and its pre-split `words`, return the
    phoneme<->word mapping, indexed correctly.

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

    def char_to_word(charpos):
        for wi, (s, e) in enumerate(word_spans):
            if s <= charpos < e:
                return wi
        return None

    text_char_to_word = [char_to_word(i) for i in range(len(text))]

    spaced_phonemes = out.phonemes
    output_pos_to_text_char = [None] * len(spaced_phonemes)
    for text_idx, m in enumerate(mappings):
        s, e = m.pos
        for p in range(s, e):
            output_pos_to_text_char[p] = text_idx

    phonemes_stripped_chars = []
    phoneme_to_word = []
    for p, ch in enumerate(spaced_phonemes):
        if ch == " ":
            continue
        text_idx = output_pos_to_text_char[p]
        wi = text_char_to_word[text_idx] if text_idx is not None else None
        phonemes_stripped_chars.append(ch)
        phoneme_to_word.append(wi)

    return {
        "words": words,
        "word_spans": word_spans,
        "phonemes": "".join(phonemes_stripped_chars),
        "phoneme_to_word": phoneme_to_word,
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


def build_combined_reference(sura_idx, tok2id, max_token_len):
    """Concatenate the whole surah's (istiaatha + all ayahs) reference
    phoneme sequences into ONE token-id list, and build a parallel list of
    "word slots" (one per real Arabic word across the whole surah) each
    carrying the list of global positions (indices into the combined
    token-id list) of the tokens that belong to that word.

    This is what makes the single whole-surah-at-once Viterbi pass possible:
    the reference position space is flat and global, so the alignment
    doesn't need to know about ayah boundaries at all -- they fall out
    naturally from which word slot each token belongs to.
    """
    from .tokenizer import tokenize_with_char_starts

    refs = build_surah_reference(sura_idx)
    combined_token_ids = []
    word_slots = []  # each: {"word","sura","aya","is_ayah_final","token_positions": [...]}

    for ref in refs:
        ids, char_starts = tokenize_with_char_starts(ref["phonemes"], tok2id, max_token_len)
        word_slot_base = len(word_slots)
        n_words_in_ref = len(ref["words"])
        for wi in range(n_words_in_ref):
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
            })
        for local_tok_idx, (tid, cs) in enumerate(zip(ids, char_starts)):
            wi = ref["phoneme_to_word"][cs] if cs < len(ref["phoneme_to_word"]) else None
            global_pos = len(combined_token_ids) + local_tok_idx
            if wi is not None:
                word_slots[word_slot_base + wi]["token_positions"].append(global_pos)
        combined_token_ids.extend(ids)

    return combined_token_ids, word_slots
