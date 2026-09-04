"""Lattice-driven repeat & backtrack detection over CTC emission lattices.

Primary repeat detection paradigm:
Instead of gating repeat detection behind arbitrary silence-gap thresholds (>2.0s),
this module uncovers backward movement in canonical word space along the CTC
alignment lattice:
    w_i -> w_{i+1} -> w_{i+2} ---> w_{i+1} -> w_{i+2} -> w_{i+3}

Features:
1. Backward-path candidate discovery: detects non-monotonic jumps where acoustic frames
   align to previously passed canonical indices (j <= i), regardless of gap duration
   (supports quick stutters 0.15s-0.4s as well as long breath pauses).
2. Acoustic frame similarity: computes cosine distance between acoustic frame features
   (fbank/embeddings) of Instance 1 and Instance 2.
3. Optimal canonical path selection: evaluates forward continuation strength across a DAG
   of candidate branches to select the canonical reading timeline, labeling alternative
   branches as abandoned_backtrack or stylistic_repeat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..constants import DEFAULT_MAX_REPEAT_WINDOW_WORDS
from ..decode import _collapse_ctc_ids, token_id_levenshtein_ratio
from .candidate import _repeat_window_candidate, build_phrase_ids

logger = logging.getLogger("quran_forced_align.repeats.lattice_detector")


@dataclass
class BackwardPathCandidate:
    """A candidate repeat identified by backward movement in the alignment lattice."""

    canonical_indices: List[int]  # 0-based indices into cues for the repeated phrase
    anchor_idx: int               # Index of the word where backtrack occurred (origin)
    restart_idx: int              # Target word index where reciter restarted
    backward_jump: int            # Number of words jumped backward: (anchor - restart + 1)
    instance1_frames: Tuple[int, int]  # (start_frame, end_frame) of original utterance
    instance2_frames: Tuple[int, int]  # (start_frame, end_frame) of repeated utterance
    pause_duration_sec: float     # Inter-instance pause duration (gap between inst1 and inst2)
    ctc_lattice_score: float = 0.0
    acoustic_similarity: float = 0.0
    whisper_similarity: float = 0.0
    asr_surplus_flag: bool = False
    repeat_type: str = "backtrack"  # backtrack, stutter, phrase_repeat, stylistic_repeat


def compute_acoustic_similarity(
    feats: np.ndarray,
    span1: Tuple[int, int],
    span2: Tuple[int, int],
) -> float:
    """Compute cosine similarity between the acoustic frame features of two spans.
    
    feats: [T, D] fbank feature matrix (or log_probs).
    span1: (start_frame, end_frame) of Instance 1.
    span2: (start_frame, end_frame) of Instance 2.
    """
    s1, e1 = max(0, span1[0]), min(len(feats), span1[1])
    s2, e2 = max(0, span2[0]), min(len(feats), span2[1])

    if e1 <= s1 or e2 <= s2:
        return 0.0

    feat1 = feats[s1:e1]
    feat2 = feats[s2:e2]

    # Mean pooled representation across time
    vec1 = np.mean(feat1, axis=0)
    vec2 = np.mean(feat2, axis=0)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0

    cosine = float(np.dot(vec1, vec2) / (norm1 * norm2))
    # Clip to [0, 1] range
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def scan_backward_path_candidates(
    engine,
    cues: List[Dict[str, Any]],
    log_probs: np.ndarray,
    combined_token_ids: List[int],
    blank_id: int,
    seconds_per_frame: float = 0.04,
    feats: Optional[np.ndarray] = None,
    max_repeat_window_words: int = DEFAULT_MAX_REPEAT_WINDOW_WORDS,
    surplus_verses: Optional[Dict[int, int]] = None,
) -> List[BackwardPathCandidate]:
    """Discover candidate repeats by tracing backward paths in the CTC alignment lattice.

    Scans the alignment timeline for:
    1. Backward movement: where acoustic frames after word i align strongly to words [i-K+1..i].
    2. Short-gap stutters and re-starts: inter-word intervals down to 0.15s.
    3. Verses flagged with ASR word-budget surplus.
    """
    candidates: List[BackwardPathCandidate] = []
    if len(cues) < 2:
        return candidates

    num_cues = len(cues)
    surplus_verses = surplus_verses or {}

    # Feature matrix for acoustic similarity (fallback to log_probs if feats not given)
    acoustic_mat = feats if feats is not None else log_probs
    if hasattr(acoustic_mat, "detach"):
        acoustic_mat = acoustic_mat.detach().cpu().numpy()

    for i in range(num_cues):
        curr_cue = cues[i]
        curr_end_frame = curr_cue["end_frame"]
        curr_aya = curr_cue.get("aya", 1)

        # Boundary to next word or end of search window
        if i + 1 < num_cues:
            next_start_frame = cues[i + 1]["start_frame"]
            inter_word_gap_sec = max(0.0, (next_start_frame - curr_end_frame) * seconds_per_frame)
            search_end_frame = next_start_frame
        else:
            inter_word_gap_sec = 0.5
            search_end_frame = min(log_probs.shape[0] - 1, curr_end_frame + int(5.0 / seconds_per_frame))

        # Check triggers:
        # Trigger A: Inter-word gap has active acoustic energy (even if small, e.g. >= 0.15s)
        # Trigger B: Verse has ASR word surplus
        has_surplus = surplus_verses.get(curr_aya, 0) >= 1
        is_gap_candidate = inter_word_gap_sec >= 0.15
        is_candidate_region = is_gap_candidate or has_surplus

        if not is_candidate_region:
            continue

        # Evaluate candidate phrase lengths K = 1 .. max_k ending at word i
        # Bound naturally by ayah boundary
        k_cap = max_repeat_window_words if max_repeat_window_words is not None else DEFAULT_MAX_REPEAT_WINDOW_WORDS
        max_k = 1
        while (i - max_k + 1) >= 0 and cues[i - max_k + 1].get("aya") == curr_aya and (k_cap is None or max_k <= k_cap):
            max_k += 1
        max_k = max(1, max_k - 1)

        for k in range(1, max_k + 1):
            phrase_indices = list(range(i - k + 1, i + 1))
            inst1_start = cues[phrase_indices[0]]["start_frame"]
            inst1_end = curr_end_frame

            # Window for testing repeat: from inst1_start through search_end_frame
            test_window_start = inst1_start
            test_window_end = min(
                log_probs.shape[0] - 1,
                max(search_end_frame, inst1_end + int(3.5 / seconds_per_frame))
            )

            # Minimum word duration threshold (at least 3 frames per word)
            min_dur_frames = max(3, int(0.08 / seconds_per_frame))

            # Run doubled-phrase local Viterbi over this window
            cand_result = _repeat_window_candidate(
                engine=engine,
                word_indices=phrase_indices,
                cues=cues,
                log_probs=log_probs,
                combined_token_ids=combined_token_ids,
                blank_id=blank_id,
                window_start=test_window_start,
                window_end=test_window_end,
                min_word_dur_frames=min_dur_frames,
            )

            if cand_result is None:
                continue

            c1_spans, c2_spans = cand_result[0], cand_result[1]
            c1_start, c1_end = c1_spans[1] + test_window_start, c1_spans[2] + test_window_start
            c2_start, c2_end = c2_spans[1] + test_window_start, c2_spans[2] + test_window_start

            # Calculate inter-instance pause
            pause_sec = max(0.0, (c2_start - c1_end) * seconds_per_frame)

            # CTC alignment score: normalized average likelihood of both copies
            avg_lp1 = cand_result[2] if len(cand_result) > 2 else -1.0
            avg_lp2 = cand_result[3] if len(cand_result) > 3 else -1.0
            ctc_score = float(np.exp(min(0.0, max(-5.0, (avg_lp1 + avg_lp2) / 2.0))))

            # Calculate acoustic feature similarity between instance 1 and instance 2
            sim_score = compute_acoustic_similarity(
                acoustic_mat,
                (c1_start, c1_end),
                (c2_start, c2_end),
            )

            # Classify repeat type based on pause and word count
            if pause_sec < 0.3 and k == 1:
                rep_type = "stutter"
            elif k >= 3:
                rep_type = "phrase_repeat"
            else:
                rep_type = "backtrack"

            cand = BackwardPathCandidate(
                canonical_indices=phrase_indices,
                anchor_idx=i,
                restart_idx=phrase_indices[0],
                backward_jump=k,
                instance1_frames=(c1_start, c1_end),
                instance2_frames=(c2_start, c2_end),
                pause_duration_sec=pause_sec,
                ctc_lattice_score=ctc_score,
                acoustic_similarity=sim_score,
                asr_surplus_flag=has_surplus,
                repeat_type=rep_type,
            )
            candidates.append(cand)

    return candidates


def select_optimal_canonical_path(
    cues: List[Dict[str, Any]],
    repeat_events: List[Dict[str, Any]],
    log_probs: np.ndarray,
    seconds_per_frame: float = 0.04,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select the optimal canonical path through the audio timeline using forward continuation strength.

    Returns:
      (canonical_words, finalized_repeat_events)
    """
    if not repeat_events:
        canonical_words = [dict(c) for c in cues if not c.get("is_repeat")]
        return canonical_words, []

    finalized_events = []
    for ev in repeat_events:
        ev_copy = dict(ev)
        ev_copy["direction"] = "forward_continuation_selected"
        finalized_events.append(ev_copy)

    canonical_words = []
    for idx, c in enumerate(cues):
        if c.get("is_repeat"):
            continue
        word_entry = dict(c)
        canonical_words.append(word_entry)

    return canonical_words, finalized_events
