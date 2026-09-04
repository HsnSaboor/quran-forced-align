import numpy as np
import pytest

from quran_forced_align.reference.surah import (
    build_bismillah_reference,
    build_surah_reference,
)
from quran_forced_align.reference.combined import build_combined_reference
from quran_forced_align.tokenizer import load_tokens, tokenize_with_char_starts
from quran_forced_align.pipeline import (
    detect_leading_bismillah,
    detect_leading_istiaatha,
    detect_leading_openings,
)
from quran_forced_align.srt import build_rich_records

from .conftest import TOKENS_PATH


@pytest.fixture(scope="module")
def tokens_tuple():
    return load_tokens(TOKENS_PATH)


class TestPreamblesReference:
    def test_surah_1_bismillah_is_ayah_1(self, tokens_tuple):
        tok2id, id2tok, blank_id, max_token_len = tokens_tuple
        # Surah 1 Bismillah reference is None because Bismillah is Aya 1:1
        b_ref = build_bismillah_reference(1)
        assert b_ref is None

        # build_surah_reference for Surah 1
        refs = build_surah_reference(1, include_istiaatha=False, include_bismillah=True)
        assert refs[0]["aya_idx"] == 1
        assert refs[0]["words"] == ["بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَـٰنِ", "ٱلرَّحِيمِ"]

        # Combined reference for Surah 1
        comb_ids, word_slots = build_combined_reference(
            1, tok2id, max_token_len, include_istiaatha=False, include_bismillah=True
        )
        assert word_slots[0]["aya"] == 1
        assert word_slots[0]["word"] == "بِسْمِ"

    def test_surah_2_to_114_bismillah_is_ayah_0(self, tokens_tuple):
        tok2id, id2tok, blank_id, max_token_len = tokens_tuple
        for sura in [2, 18, 36, 67, 114]:
            b_ref = build_bismillah_reference(sura)
            assert b_ref is not None
            assert b_ref["sura_idx"] == sura
            assert b_ref["aya_idx"] == 0
            assert b_ref["words"] == ["بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَـٰنِ", "ٱلرَّحِيمِ"]
            assert len(b_ref["words"]) == 4

            # Combined without Bismillah
            _, slots_no_bsm = build_combined_reference(
                sura, tok2id, max_token_len, include_istiaatha=False, include_bismillah=False
            )
            assert slots_no_bsm[0]["aya"] == 1

            # Combined with Bismillah
            _, slots_with_bsm = build_combined_reference(
                sura, tok2id, max_token_len, include_istiaatha=False, include_bismillah=True
            )
            assert slots_with_bsm[0]["aya"] == 0
            assert slots_with_bsm[0]["word"] == "بِسْمِ"
            assert slots_with_bsm[3]["word"] == "ٱلرَّحِيمِ"
            assert slots_with_bsm[4]["aya"] == 1
            assert slots_with_bsm[4]["word"] == slots_no_bsm[0]["word"]

    def test_surah_9_has_no_bismillah(self, tokens_tuple):
        tok2id, id2tok, blank_id, max_token_len = tokens_tuple
        b_ref = build_bismillah_reference(9)
        assert b_ref is None

        refs = build_surah_reference(9, include_istiaatha=False, include_bismillah=True)
        assert refs[0]["aya_idx"] == 1
        assert not any(r["aya_idx"] == 0 for r in refs)

        _, slots = build_combined_reference(
            9, tok2id, max_token_len, include_istiaatha=False, include_bismillah=True
        )
        assert slots[0]["aya"] == 1

    def test_istiaatha_and_bismillah_both_preamble(self, tokens_tuple):
        tok2id, id2tok, blank_id, max_token_len = tokens_tuple
        _, slots = build_combined_reference(
            2, tok2id, max_token_len, include_istiaatha=True, include_bismillah=True
        )
        # Isti'adha words (5) + Bismillah words (4) + Ayah 1 (1)
        assert slots[0]["word"] == "أَعُوذُ"
        assert slots[0]["aya"] == 0
        assert slots[4]["word"] == "ٱلرَّجِيمِ"
        assert slots[4]["aya"] == 0
        assert slots[5]["word"] == "بِسْمِ"
        assert slots[5]["aya"] == 0
        assert slots[8]["word"] == "ٱلرَّحِيمِ"
        assert slots[8]["aya"] == 0
        assert slots[9]["aya"] == 1
        assert slots[9]["word"] == "الٓمٓ"


class TestAcousticProbe:
    def test_detect_leading_openings(self, tokens_tuple):
        tok2id, id2tok, blank_id, max_token_len = tokens_tuple
        V = max(id2tok.keys()) + 1

        # 1. Synthesize log_probs with Isti'adha + Bismillah tokens
        ist_toks = ['ءَ', 'عُ', 'ۥۥ', 'ذُ', 'بِ', 'للَ', 'اا', 'هِ', 'مِ', 'نَ', 'ششَ', 'ي', 'طَ', 'اا', 'نِ', 'ررَ', 'جِ', 'ۦۦۦۦ', 'م']
        bsm_toks = ['بِ', 'س', 'مِ', 'للَ', 'اا', 'هِ', 'ررَ', 'ح', 'مَ', 'اا', 'نِ', 'ررَ', 'حِ', 'ۦۦۦۦ', 'م']
        
        T = 200
        log_probs = np.full((T, V), -10.0, dtype=np.float32)
        
        t = 0
        for tok in ist_toks:
            if tok in tok2id:
                tid = tok2id[tok]
                log_probs[t:t+3, tid] = 0.0
                t += 4
        for tok in bsm_toks:
            if tok in tok2id:
                tid = tok2id[tok]
                log_probs[t:t+3, tid] = 0.0
                t += 4

        has_ist, has_bsm = detect_leading_openings(log_probs, id2tok)
        assert has_ist is True
        assert has_bsm is True
        assert detect_leading_istiaatha(log_probs, id2tok) is True
        assert detect_leading_bismillah(log_probs, id2tok) is True

    def test_detect_leading_bismillah_only(self, tokens_tuple):
        tok2id, id2tok, blank_id, max_token_len = tokens_tuple
        V = max(id2tok.keys()) + 1

        bsm_toks = ['بِ', 'س', 'مِ', 'للَ', 'اا', 'هِ', 'ررَ', 'ح', 'مَ', 'اا', 'نِ', 'ررَ', 'حِ', 'ۦۦۦۦ', 'م']
        T = 100
        log_probs = np.full((T, V), -10.0, dtype=np.float32)
        
        t = 0
        for tok in bsm_toks:
            if tok in tok2id:
                tid = tok2id[tok]
                log_probs[t:t+3, tid] = 0.0
                t += 4

        has_ist, has_bsm = detect_leading_openings(log_probs, id2tok)
        assert has_ist is False
        assert has_bsm is True
        assert detect_leading_istiaatha(log_probs, id2tok) is False
        assert detect_leading_bismillah(log_probs, id2tok) is True


class TestOutputStripping:
    def test_surah_1_retains_bismillah_in_rich_records(self, tokens_tuple):
        tok2id, id2tok, blank_id, max_token_len = tokens_tuple
        comb_ids, slots = build_combined_reference(1, tok2id, max_token_len)
        
        # Mock cue records matching slots
        mock_cues = []
        for i, s in enumerate(slots):
            mock_cues.append({
                **s,
                "start_frame": i * 10,
                "end_frame": (i + 1) * 10 - 1,
                "is_repeat": False,
                "low_confidence": False,
                "token_frame_spans": [(i * 10, (i + 1) * 10 - 1) for _ in s["token_positions"]],
            })
            
        records = build_rich_records(mock_cues, 0.04, comb_ids, id2tok, strip_istiaatha=True)
        # Surah 1 Bismillah has aya == 1, so all words are retained
        assert len(records) == len(slots)
        assert records[0]["aya"] == 1
        assert records[0]["word"] == "بِسْمِ"
        assert records[0]["start"] == 0.0

    def test_surah_2_omits_bismillah_and_istiaatha_in_rich_records(self, tokens_tuple):
        tok2id, id2tok, blank_id, max_token_len = tokens_tuple
        comb_ids, slots = build_combined_reference(
            2, tok2id, max_token_len, include_istiaatha=True, include_bismillah=True
        )
        
        mock_cues = []
        for i, s in enumerate(slots):
            mock_cues.append({
                **s,
                "start_frame": i * 10,
                "end_frame": (i + 1) * 10 - 1,
                "is_repeat": False,
                "low_confidence": False,
                "token_frame_spans": [(i * 10, (i + 1) * 10 - 1) for _ in s["token_positions"]],
            })
            
        # Preamble stripping: strip_istiaatha=True strips all aya == 0 preambles
        records = build_rich_records(mock_cues, 0.04, comb_ids, id2tok, strip_istiaatha=True)
        # All aya == 0 preambles are omitted
        assert all(r["aya"] >= 1 for r in records)
        # Output starts with Ayah 1, Word 1
        assert records[0]["aya"] == 1
        assert records[0]["word"] == "الٓمٓ"
        # Start time of first word reflects its actual aligned position
        assert records[0]["start"] == pytest.approx(9 * 10 * 0.04)  # 9 preamble words * 10 frames * 0.04s = 3.6s
