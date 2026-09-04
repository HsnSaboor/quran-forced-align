"""Word-level cue extraction + lattice-driven repeat detection.

Provides:
- extract_word_frame_spans, token_frame_spans: frame-level span extraction.
- detect_and_fix_repeats: heuristic repeat detector (backward compatible).
- BackwardPathCandidate, scan_backward_path_candidates, select_optimal_canonical_path:
  lattice-driven backward-path detector.
- RepeatCandidateEvaluator, WhisperVerifier, ReviewQueueExporter:
  multi-feature calibration and review queue management.
"""
from .candidate_evaluator import (
    RepeatCandidateEvaluator,
    ReviewQueueExporter,
    WhisperVerifier,
    normalize_arabic_for_asr,
    phrase_alignment_similarity,
)
from .detection import detect_and_fix_repeats
from .lattice_detector import (
    BackwardPathCandidate,
    compute_acoustic_similarity,
    scan_backward_path_candidates,
    select_optimal_canonical_path,
)
from .spans import extract_word_frame_spans, token_frame_spans

__all__ = [
    "token_frame_spans",
    "extract_word_frame_spans",
    "detect_and_fix_repeats",
    "BackwardPathCandidate",
    "compute_acoustic_similarity",
    "scan_backward_path_candidates",
    "select_optimal_canonical_path",
    "RepeatCandidateEvaluator",
    "WhisperVerifier",
    "ReviewQueueExporter",
    "normalize_arabic_for_asr",
    "phrase_alignment_similarity",
]
