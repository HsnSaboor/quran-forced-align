import quran_transcript as qt

from .text import build_text_reference


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
