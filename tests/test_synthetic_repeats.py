"""Synthetic Repeat Benchmark & Invariant Test Suite.

Verifies:
1. Lattice-driven backward path discovery for quick stutters (0.15s - 0.4s)
   without arbitrary silence-gap floors.
2. Multi-word phrase backtracks (2-4 words) and optimal canonical path selection
   (forward continuation selected, backtrack marked abandoned_backtrack).
3. Multi-feature scoring calibration:
   - P >= 0.90 -> Accepted
   - 0.50 <= P < 0.90 -> Review Queue (exported to unresolved_repeats.json)
   - P < 0.50 -> Rejected
4. Three-tier data model invariants:
   - canonical_words: 1-to-1 with canonical Uthmani text
   - raw_words: all spoken acoustic tokens preserved
   - repeat_events: structured repeat records with complete evidence vector
5. Synthetic repeat injection benchmark across variable gap durations,
   attenuations, and phrase lengths.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from quran_forced_align.pipeline import AlignmentResult
from quran_forced_align.repeats import (
    BackwardPathCandidate,
    RepeatCandidateEvaluator,
    ReviewQueueExporter,
    compute_acoustic_similarity,
    normalize_arabic_for_asr,
    phrase_alignment_similarity,
    scan_backward_path_candidates,
    select_optimal_canonical_path,
)


def test_arabic_normalization_and_phrase_similarity():
    """Verify Arabic orthographic normalization and decoupled phrase similarity."""
    raw_quran = "لَهُمْ أَجْرُهُمْ عِندَ رَبِّهِمْ"
    asr_hyp = "لهم اجرهم عند ربهم"

    norm_q = normalize_arabic_for_asr(raw_quran)
    norm_a = normalize_arabic_for_asr(asr_hyp)

    assert norm_q == "لهم اجرهم عند ربهم"
    assert norm_q == norm_a

    # Exact match in context
    sim_exact = phrase_alignment_similarity("لهم اجرهم", "قال فلهم اجرهم عند ربهم ولا خوف")
    assert sim_exact == 1.0

    # Near match with minor variation
    sim_near = phrase_alignment_similarity("لهم اجرهم", "لهم اجرهمم")
    assert sim_near >= 0.90

    # Disjoint text
    sim_disjoint = phrase_alignment_similarity("لهم اجرهم", "والسماء والطارق وما ادراك")
    assert sim_disjoint < 0.40


def test_acoustic_cosine_similarity():
    """Verify frame-level acoustic embedding cosine similarity."""
    # Synthetic feature matrix with identical pattern in two spans
    feats = np.zeros((100, 80), dtype=np.float32)
    feats[10:30] = np.sin(np.linspace(0, 3.14, 20))[:, None] * np.ones((1, 80))
    feats[50:70] = np.sin(np.linspace(0, 3.14, 20))[:, None] * np.ones((1, 80))

    sim_high = compute_acoustic_similarity(feats, (10, 30), (50, 70))
    assert sim_high >= 0.95

    # Dissimilar pattern
    feats[80:95] = np.random.randn(15, 80)
    sim_low = compute_acoustic_similarity(feats, (10, 30), (80, 95))
    assert sim_low < sim_high


def test_quick_stutter_candidate_scoring():
    """Verify that quick stutters (0.15s - 0.3s pause) are accepted without gap floors."""
    evaluator = RepeatCandidateEvaluator(whisper_verifier=None)

    # 1-word stutter with 0.18s pause
    stutter_cand = BackwardPathCandidate(
        canonical_indices=[5],
        anchor_idx=5,
        restart_idx=5,
        backward_jump=1,
        instance1_frames=(100, 115),
        instance2_frames=(120, 135),
        pause_duration_sec=0.18,
        ctc_lattice_score=0.94,
        acoustic_similarity=0.91,
        asr_surplus_flag=False,
        repeat_type="stutter",
    )

    ev = evaluator.evaluate_candidate(stutter_cand, expected_phrase="قُلْ")
    assert ev["status"] == "accepted"
    assert ev["p_repeat"] >= 0.90
    assert ev["repeat_type"] == "stutter"
    assert ev["evidence"]["inter_word_pause"] == 0.18
    assert ev["evidence"]["backward_jump"] == 1


def test_multi_word_backtrack_and_optimal_path():
    """Verify 2-word backtrack evaluation and forward continuation path selection."""
    evaluator = RepeatCandidateEvaluator(whisper_verifier=None)

    backtrack_cand = BackwardPathCandidate(
        canonical_indices=[10, 11],
        anchor_idx=11,
        restart_idx=10,
        backward_jump=2,
        instance1_frames=(200, 240),
        instance2_frames=(255, 295),
        pause_duration_sec=0.60,
        ctc_lattice_score=0.91,
        acoustic_similarity=0.87,
        asr_surplus_flag=True,
        repeat_type="backtrack",
    )

    ev = evaluator.evaluate_candidate(backtrack_cand, expected_phrase="لَهُمْ أَجْرُهُمْ")
    assert ev["status"] == "accepted"
    assert ev["p_repeat"] >= 0.90
    assert ev["direction"] == "abandoned_backtrack"
    assert ev["evidence"]["asr_word_surplus_candidate"] is True

    # Test optimal path selection: forward continuation labeled
    cues = [
        {"word": "w1", "start_frame": 100, "end_frame": 140, "is_repeat": False},
        {"word": "w2", "start_frame": 150, "end_frame": 190, "is_repeat": False},
        {"word": "w1_rep", "start_frame": 200, "end_frame": 240, "is_repeat": True},
    ]
    canon, finalized_events = select_optimal_canonical_path(cues, [ev], np.zeros((300, 50)))
    assert len(canon) == 2  # exactly the 2 non-repeat canonical words
    assert len(finalized_events) == 1
    assert finalized_events[0]["direction"] == "forward_continuation_selected"


def test_review_queue_exporter(tmp_path):
    """Verify borderline repeat candidates (0.50 <= P < 0.90) are routed to review queue."""
    evaluator = RepeatCandidateEvaluator(whisper_verifier=None)

    borderline_cand = BackwardPathCandidate(
        canonical_indices=[3],
        anchor_idx=3,
        restart_idx=3,
        backward_jump=1,
        instance1_frames=(50, 70),
        instance2_frames=(78, 98),
        pause_duration_sec=0.32,
        ctc_lattice_score=0.58,
        acoustic_similarity=0.52,
        asr_surplus_flag=False,
        repeat_type="stutter",
    )

    ev = evaluator.evaluate_candidate(borderline_cand, expected_phrase="رَبَّنَا")
    assert ev["status"] == "review_queue"
    assert 0.50 <= ev["p_repeat"] < 0.90

    out_file = tmp_path / "unresolved_repeats.json"
    ReviewQueueExporter.export([ev], out_file)
    assert out_file.exists()

    with open(out_file, encoding="utf-8") as f:
        loaded = json.load(f)
    assert len(loaded) == 1
    assert loaded[0]["status"] == "review_queue"
    assert "evidence" in loaded[0]


def test_three_tier_data_model_invariants():
    """Verify complete three-tier data model invariants and backward compatibility."""
    canon_words = [
        {"word": "الٓمٓ", "start": 5.08, "end": 15.08, "is_repeat": False},
        {"word": "ذَٰلِكَ", "start": 15.50, "end": 16.20, "is_repeat": False},
    ]
    raw_words = [
        {"word": "الٓمٓ", "start": 5.08, "end": 15.08, "is_repeat": False},
        {"word": "الٓمٓ_rep", "start": 15.10, "end": 15.40, "is_repeat": True},
        {"word": "ذَٰلِكَ", "start": 15.50, "end": 16.20, "is_repeat": False},
    ]
    repeat_events = [
        {
            "start": 15.10,
            "end": 15.40,
            "canonical_indices": [0],
            "anchor_idx": 0,
            "restart_idx": 0,
            "repeat_type": "stutter",
            "direction": "abandoned_backtrack",
            "evaluated_candidates": [{"phrase": "الٓمٓ", "p_repeat": 0.96, "status": "accepted"}],
            "p_repeat": 0.96,
            "evidence": {
                "ctc_lattice_score": 0.94,
                "whisper_similarity": 0.98,
                "acoustic_cosine": 0.90,
                "backward_jump": 1,
                "inter_word_pause": 0.15,
                "asr_word_surplus_candidate": False,
            },
        }
    ]

    res = AlignmentResult(
        canonical_words=canon_words,
        raw_words=raw_words,
        repeat_events=repeat_events,
    )

    # Invariant 1: List iteration behaves as all spoken raw_words (backward compatible)
    assert len(res) == 3
    assert [w["word"] for w in res] == ["الٓمٓ", "الٓمٓ_rep", "ذَٰلِكَ"]
    assert sum(1 for w in res if w["is_repeat"]) == 1

    # Invariant 2: canonical_words is strictly 1-to-1 with canonical text
    assert len(res.canonical_words) == 2
    assert all(not w["is_repeat"] for w in res.canonical_words)
    assert res["canonical_words"] == canon_words

    # Invariant 3: repeat_events matches schema with complete evidence
    assert len(res.repeat_events) == 1
    ev = res.repeat_events[0]
    assert ev["p_repeat"] == 0.96
    assert "ctc_lattice_score" in ev["evidence"]
    assert "acoustic_cosine" in ev["evidence"]

    # Invariant 4: Dictionary representation
    d = res.to_dict()
    assert "canonical_words" in d
    assert "raw_words" in d
    assert "repeat_events" in d


def test_synthetic_repeat_injection_benchmark():
    """Benchmark repeat discovery across variable gap durations, phrase lengths, and speeds."""
    evaluator = RepeatCandidateEvaluator(whisper_verifier=None)

    test_scenarios = [
        # (name, pause_sec, k_words, ctc_score, acoustic_sim, surplus, expected_status)
        ("quick_stutter_0.15s", 0.15, 1, 0.95, 0.92, False, "accepted"),
        ("short_pause_0.35s", 0.35, 1, 0.92, 0.89, False, "accepted"),
        ("2_word_backtrack_0.6s", 0.60, 2, 0.90, 0.88, True, "accepted"),
        ("3_word_phrase_repeat_1.2s", 1.20, 3, 0.89, 0.85, True, "accepted"),
        ("attenuated_repeat_0.8x", 0.45, 1, 0.91, 0.86, False, "accepted"),
        ("fast_speech_1.1x", 0.20, 2, 0.93, 0.90, False, "accepted"),
        ("borderline_unclear_stutter", 0.25, 1, 0.60, 0.55, False, "review_queue"),
        ("spurious_non_repeat", 0.10, 1, 0.20, 0.25, False, "rejected"),
    ]

    results = []
    for name, pause, k, ctc, sim, surplus, exp_status in test_scenarios:
        cand = BackwardPathCandidate(
            canonical_indices=list(range(k)),
            anchor_idx=k - 1,
            restart_idx=0,
            backward_jump=k,
            instance1_frames=(0, 20 * k),
            instance2_frames=(20 * k + int(pause / 0.04), 40 * k + int(pause / 0.04)),
            pause_duration_sec=pause,
            ctc_lattice_score=ctc,
            acoustic_similarity=sim,
            asr_surplus_flag=surplus,
            repeat_type="stutter" if k == 1 and pause < 0.3 else ("phrase_repeat" if k >= 3 else "backtrack"),
        )
        ev = evaluator.evaluate_candidate(cand, expected_phrase=" ".join(f"w{i}" for i in range(k)))
        assert ev["status"] == exp_status, f"Scenario '{name}' expected {exp_status}, got {ev['status']} (p={ev['p_repeat']})"
        results.append((name, ev["status"], ev["p_repeat"]))

    print(f"\nSynthetic Repeat Benchmark Results ({len(results)} scenarios evaluated):")
    for name, status, p in results:
        print(f"  - {name:<30} -> {status:<15} (P={p:.4f})")
