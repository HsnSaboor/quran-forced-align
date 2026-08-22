import gzip
import os
import pickle
import quran_transcript as qt

from .text import build_text_reference

_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "quran_refs.pkl.gz")
_LOADED_CACHE = None


def _get_precomputed_cache():
    global _LOADED_CACHE
    if _LOADED_CACHE is None:
        if os.path.isfile(_CACHE_FILE):
            try:
                with gzip.open(_CACHE_FILE, "rb") as f:
                    _LOADED_CACHE = pickle.load(f)
            except Exception:
                _LOADED_CACHE = {}
        else:
            _LOADED_CACHE = {}
    return _LOADED_CACHE


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
    cache = _get_precomputed_cache()
    if sura_idx in cache:
        cached_refs = cache[sura_idx]
        if not include_istiaatha and cached_refs and cached_refs[0].get("aya_idx") == 0:
            return cached_refs[1:]
        return cached_refs

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
