import pytest
from quran_forced_align.corrections import (
    align_spoken_to_canonical,
    normalize_quran_text,
    LearnedCorrectionsRegistry,
)

def test_normalize_quran_text():
    assert normalize_quran_text("ثُمَّ") == normalize_quran_text("ثُمَّ")
    assert normalize_quran_text("مُصَدِّقًۢا") == normalize_quran_text("مُصَدِّقًۭا")

def test_align_spoken_to_canonical_simple():
    canon = ["بَلَىٰ", "مَنْ", "أَسْلَمَ"]
    spoken = ["بَلَىٰ", "مَنْ", "أَسْلَمَ"]
    mapping = align_spoken_to_canonical(canon, spoken)
    assert mapping == [1, 2, 3]

def test_align_spoken_to_canonical_with_repeat():
    # Ayah 112 repeat scenario: words 1, 2, 3, 2, 3, 4
    canon = ["w1", "w2", "w3", "w4"]
    spoken = ["w1", "w2", "w3", "w2", "w3", "w4"]
    mapping = align_spoken_to_canonical(canon, spoken)
    assert mapping == [1, 2, 3, 2, 3, 4]

def test_learned_corrections_registry(tmp_path):
    reg_path = tmp_path / "corrections.json"
    reg = LearnedCorrectionsRegistry(reg_path)
    assert reg.get_corrections("mansour-al-salmi", 2, 109) == []
    
    reg.register_repeat(
        reciter="mansour-al-salmi",
        surah_id=2,
        ayah_id=109,
        word_indices=[11, 12],
        phrase="كُفَّارًا حَسَدًۭا",
        start_ms=2604560,
        end_ms=2606000,
    )
    
    reps = reg.get_corrections("mansour-al-salmi", 2, 109)
    assert len(reps) == 1
    assert reps[0]["words"] == [11, 12]
    assert reps[0]["phrase"] == "كُفَّارًا حَسَدًۭا"
    
    # Reload from disk
    reg2 = LearnedCorrectionsRegistry(reg_path)
    reps2 = reg2.get_corrections("mansour-al-salmi", 2, 109)
    assert len(reps2) == 1
    assert reps2[0]["words"] == [11, 12]
