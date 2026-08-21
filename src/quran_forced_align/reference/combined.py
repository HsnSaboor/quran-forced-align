import copy
import gzip
import os
import pickle
from functools import lru_cache

from ..tokenizer import tokenize_with_char_starts
from .boundary import _boundary_bridge_rules
from .surah import build_surah_reference

_REFS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "references")


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
    s_file = os.path.join(_REFS_DIR, f"{sura_idx}.pkl")
    if os.path.isfile(s_file):
        try:
            with open(s_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    refs = _cached_surah_reference(sura_idx)
    # Deep-copy only the mutable fields (char_info dicts get boundary_tajweed_rules
    # mutated below). The phonemes/word_spans/phoneme_to_word/phoneme_to_char are
    # read-only and safe to share.
    refs = [
        {**ref, "char_info": copy.deepcopy(ref["char_info"])}
        for ref in refs
    ]
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
