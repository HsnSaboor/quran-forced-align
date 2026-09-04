import copy
import gzip
import os
import pickle
import quran_transcript as qt

from .text import build_text_reference

_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "quran_refs.pkl.gz")
_LOADED_CACHE = None
_BASE_BISMILLAH_REF = None


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


def build_bismillah_reference(sura_idx):
    """Build Aya 0 reference for Bismillah preamble (Surahs 2-114 except 9).
    For Surah 1 (Al-Fatihah), Bismillah is Aya 1:1, so this returns None.
    For Surah 9 (At-Tawbah), there is no Bismillah in the Quran, so this returns None.
    """
    global _BASE_BISMILLAH_REF
    if sura_idx == 1 or sura_idx == 9:
        return None
    if _BASE_BISMILLAH_REF is None:
        text = "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"
        words = text.split()
        _BASE_BISMILLAH_REF = build_text_reference(text, words, 0, 0)
    return {
        **_BASE_BISMILLAH_REF,
        "sura_idx": sura_idx,
        "aya_idx": 0,
        "char_info": copy.deepcopy(_BASE_BISMILLAH_REF["char_info"]),
    }


def build_surah_reference(sura_idx, include_istiaatha=False, include_bismillah=False):
    cache = _get_precomputed_cache()
    if sura_idx in cache:
        cached_entry = cache[sura_idx]
        if isinstance(cached_entry, list):
            istiaatha_cached = cached_entry[0] if cached_entry and cached_entry[0].get("aya_idx") == 0 else None
            ayah_cached = cached_entry[1:] if cached_entry and cached_entry[0].get("aya_idx") == 0 else cached_entry
        elif isinstance(cached_entry, dict):
            istiaatha_cached = cached_entry.get("istiaatha")
            ayah_cached = cached_entry.get("ayahs", [])
        else:
            istiaatha_cached = None
            ayah_cached = []

        refs = []
        if include_istiaatha:
            ist = istiaatha_cached or build_istiaatha_reference(sura_idx)
            if ist:
                refs.append(ist)
        if include_bismillah and sura_idx not in (1, 9):
            bsm = build_bismillah_reference(sura_idx)
            if bsm:
                refs.append(bsm)
        refs.extend(ayah_cached)
        return refs

    aya = qt.Aya(sura_idx, 1)
    d = aya.get()
    n_ayat = d.num_ayat_in_sura
    refs = []
    if include_istiaatha:
        preamble = build_istiaatha_reference(sura_idx)
        if preamble:
            refs.append(preamble)
    if include_bismillah and sura_idx not in (1, 9):
        bsm = build_bismillah_reference(sura_idx)
        if bsm:
            refs.append(bsm)
    for aya_idx in range(1, n_ayat + 1):
        refs.append(build_ayah_reference(sura_idx, aya_idx))
    return refs
