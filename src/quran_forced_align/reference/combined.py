import copy
import os
import pickle
from functools import lru_cache

from ..tokenizer import tokenize_with_char_starts
from .boundary import _boundary_bridge_rules
from .surah import build_surah_reference

_REFS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "references")

# Process-level in-memory cache for 0.00ms Stage 1 reference retrieval
_COMBINED_CACHE = {}


# Cache the expensive quran_phonetizer calls — Quran text is 100% static and
# deterministic, so build_surah_reference(sura_idx) returns the identical
# result on every call for a given surah. Caching all 114 surahs uses ~50MB
# of RAM and eliminates ~25s of sequential Python regex/Unicode processing
# on Al-Baqarah (286 ayahs × ~80ms each).
@lru_cache(maxsize=128)
def _cached_surah_reference(sura_idx):
    """Cache the raw surah reference data (phonemes, word spans, char_info).
    Returns a tuple of frozen dicts for hashability — callers must deep-copy
    mutable fields before mutation."""
    return build_surah_reference(sura_idx)


def build_combined_reference(sura_idx, tok2id, max_token_len, include_boundary_tajweed=True, include_istiaatha=True):
    """Concatenate the whole surah's (istiaatha + all ayahs) reference
    phoneme sequences into ONE token-id list, and build a parallel list of
    "word slots" (one per real Arabic word across the whole surah) each
    carrying the list of global positions (indices into the combined
    token-id list) of the tokens that belong to that word, plus per-token
    character attribution and per-character tajweed/silent-letter data
    needed for the letter/phoneme-tier output (see `cells.py`).

    If `include_istiaatha=False`, strips the Isti'adha (Aya 0) preamble so
    recordings without Isti'adha align cleanly with zero leading drift.
    """
    cache_key = (sura_idx, bool(include_boundary_tajweed), bool(include_istiaatha))
    if cache_key in _COMBINED_CACHE:
        return _COMBINED_CACHE[cache_key]
    s_file = os.path.join(_REFS_DIR, f"{sura_idx}.pkl")
    res = None
    if os.path.isfile(s_file):
        try:
            with open(s_file, "rb") as f:
                res = pickle.load(f)
        except Exception:
            res = None

    if res is None:
        refs = _cached_surah_reference(sura_idx)
        # Deep-copy only the mutable fields (char_info dicts get boundary_tajweed_rules
        # mutated below). The phonemes/word_spans/phoneme_to_word/phoneme_to_char are
        # read-only and safe to share.
        refs = [
            {**ref, "char_info": copy.deepcopy(ref["char_info"])}
            for ref in refs
        ]
        combined_token_ids = []
        word_slots = []

        for ref in refs:
            ids, char_starts = tokenize_with_char_starts(ref["phonemes"], tok2id, max_token_len)
            word_slot_base = len(word_slots)
            n_words_in_ref = len(ref["words"])
            for wi, (word_start, word_end) in enumerate(ref["word_spans"]):
                word_slots.append({
                    "word": ref["words"][wi],
                    "sura": ref["sura_idx"],
                    "aya": ref["aya_idx"],
                    "is_ayah_final": wi == n_words_in_ref - 1,
                    "token_positions": [],
                    "token_char_idx": [],
                    "letters": ref["char_info"][word_start:word_end],
                })
            for local_tok_idx, cs in enumerate(char_starts):
                wi = ref["phoneme_to_word"][cs] if cs < len(ref["phoneme_to_word"]) else None
                char_idx = ref["phoneme_to_char"][cs] if cs < len(ref["phoneme_to_char"]) else None
                global_pos = len(combined_token_ids) + local_tok_idx
                if wi is not None:
                    slot = word_slots[word_slot_base + wi]
                    slot["token_positions"].append(global_pos)
                    word_start = ref["word_spans"][wi][0]
                    slot["token_char_idx"].append(char_idx - word_start)
            combined_token_ids.extend(ids)

        if include_boundary_tajweed:
            for i in range(len(word_slots) - 1):
                if not word_slots[i]["is_ayah_final"]:
                    continue
                prev_extra, next_extra = _boundary_bridge_rules(word_slots[i]["word"], word_slots[i + 1]["word"])
                for char_idx, rules in prev_extra:
                    word_slots[i]["letters"][char_idx]["boundary_tajweed_rules"] = rules
                for char_idx, rules in next_extra:
                    word_slots[i + 1]["letters"][char_idx]["boundary_tajweed_rules"] = rules

        res = (combined_token_ids, word_slots)

    if not include_istiaatha:
        comb_ids, w_slots = res
        istiaatha_words = [w for w in w_slots if w.get("aya") == 0]
        if istiaatha_words:
            first_real_word = [w for w in w_slots if w.get("aya") > 0][0]
            shift = first_real_word["token_positions"][0]
            clean_tokens = comb_ids[shift:]
            clean_word_slots = []
            for w in w_slots:
                if w.get("aya") > 0:
                    clean_word_slots.append({
                        **w,
                        "token_positions": [p - shift for p in w["token_positions"]]
                    })
            res = (clean_tokens, clean_word_slots)

    _COMBINED_CACHE[cache_key] = res
    return res
