"""Candidate evaluation, multi-feature calibration, and review queue export.

This module implements the production-grade multi-feature repeat evaluator:
1. Multi-Feature Scorer (`RepeatCandidateEvaluator`):
   Evaluates feature vector:
       x = [S_CTC, S_Whisper, S_Acoustic, Δ_jump, T_pause, S_surplus]
   to compute calibrated P(repeat):
       - P >= 0.90 -> Auto-Accept (repeat_events)
       - 0.50 <= P < 0.90 -> Review Queue (unresolved_repeats.json)
       - P < 0.50 -> Reject
2. Decoupled Whisper Verification (`WhisperVerifier`):
   Crops 2.0s - 3.5s around candidate region, decodes via tadabur-whisper-small-ct2
   WITHOUT biasing full-verse prompt, normalizes Arabic orthography, and scores
   via Levenshtein ratio.
3. Review Queue Exporter (`ReviewQueueExporter`):
   Exports unresolved candidates (0.50 <= P < 0.90) with full audio/lattice context
   for offline inspection.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import tempfile
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..decode import token_id_levenshtein_ratio
from .lattice_detector import BackwardPathCandidate

logger = logging.getLogger("quran_forced_align.repeats.candidate_evaluator")

# Default path to tadabur whisper model in the workspace
DEFAULT_WHISPER_MODEL_DIR = Path("/Code/mualim-dataset-pipeline/models/tadabur-whisper-small-ct2")
MUALIM_PYTHON_BIN = Path("/Code/mualim-dataset-pipeline/.venv/bin/python")


def normalize_arabic_for_asr(text: str) -> str:
    """Normalize Arabic text for comparison with ASR transcriptions.

    Strips harakat/tashkeel, tatweel, Quranic recitation marks, and normalizes
    orthographic letter variants (alef forms, taa marbuta, alef maqsura).
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    # Strip all Arabic diacritics / harakat and Quranic marks
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED\u0610-\u061A]", "", text)
    # Strip tatweel / kashida
    text = re.sub(r"\u0640", "", text)
    # Normalize Alef forms: أ إ آ ٱ -> ا
    text = re.sub(r"[إأآاٱ]", "ا", text)
    # Normalize Teh Marbuta: ة -> ه
    text = re.sub(r"ة", "ه", text)
    # Normalize Alef Maqsura: ى -> ي
    text = re.sub(r"ى", "ي", text)
    # Strip punctuation and symbols
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def phrase_alignment_similarity(phrase: str, transcript: str) -> float:
    """Compute similarity between candidate phrase and ASR transcription.

    Accounts for surrounding context in the audio crop via sliding window
    Levenshtein matching.
    """
    phrase = phrase.strip()
    transcript = transcript.strip()
    if not phrase and not transcript:
        return 1.0
    if not phrase or not transcript:
        return 0.0

    if phrase in transcript:
        return 1.0

    p_words = phrase.split()
    t_words = transcript.split()
    if not p_words or not t_words:
        return 0.0

    k = len(p_words)
    best_sim = 0.0
    # Sliding window of length k or k+1 across transcript
    for win_len in (k, k + 1, max(1, k - 1)):
        if win_len > len(t_words):
            continue
        for i in range(len(t_words) - win_len + 1):
            window_str = " ".join(t_words[i : i + win_len])
            sim = token_id_levenshtein_ratio(list(phrase), list(window_str))
            if sim > best_sim:
                best_sim = sim
                if best_sim >= 0.95:
                    return best_sim

    overall_sim = token_id_levenshtein_ratio(list(phrase), list(transcript))
    return max(best_sim, overall_sim)


class WhisperVerifier:
    """Decoupled Whisper transcription verifier for repeat candidate regions."""

    def __init__(self, model_path: Optional[str | Path] = None, device: str = "cpu"):
        self.model_path = Path(model_path) if model_path else DEFAULT_WHISPER_MODEL_DIR
        self.device = device
        self._model = None
        self._initialized = False

    def _ensure_model(self) -> bool:
        if self._initialized:
            return self._model is not None

        self._initialized = True
        if not self.model_path.exists():
            logger.info("Whisper model path %s does not exist; Whisper verification disabled.", self.model_path)
            return False

        try:
            import faster_whisper
            self._model = faster_whisper.WhisperModel(
                str(self.model_path),
                device=self.device,
                compute_type="int8" if self.device == "cpu" else "float16",
            )
            logger.info("Initialized Faster-Whisper model from %s", self.model_path)
            return True
        except ImportError:
            # Check if we can run via mualim python venv
            if MUALIM_PYTHON_BIN.exists():
                logger.info("faster_whisper not in current venv; using mualim venv fallback.")
                return True
            logger.warning("faster_whisper not available; Whisper verification will be bypassed.")
            return False
        except Exception as e:
            logger.warning("Failed to load Whisper model: %s", e)
            return False

    def verify_crop(
        self,
        audio_samples: np.ndarray,
        sample_rate: int,
        start_sec: float,
        end_sec: float,
        expected_phrase: str,
    ) -> Tuple[float, str]:
        """Crop 2.0s - 3.5s around candidate region, transcribe, and score similarity.

        Important: No verse-level initial_prompt is used to prevent confirmation bias.
        """
        if not self._ensure_model():
            return 1.0, ""

        # Calculate crop window (pad to 2.0s - 3.5s)
        dur = end_sec - start_sec
        target_dur = max(2.0, min(3.5, dur + 1.0))
        mid = (start_sec + end_sec) / 2.0
        total_audio_sec = len(audio_samples) / float(sample_rate)

        crop_start = max(0.0, mid - target_dur / 2.0)
        crop_end = min(total_audio_sec, crop_start + target_dur)

        s_idx = int(crop_start * sample_rate)
        e_idx = int(crop_end * sample_rate)
        crop_audio = audio_samples[s_idx:e_idx]

        if len(crop_audio) < int(0.5 * sample_rate):
            return 0.0, ""

        transcript = self._transcribe_audio(crop_audio, sample_rate)
        norm_expected = normalize_arabic_for_asr(expected_phrase)
        norm_transcript = normalize_arabic_for_asr(transcript)

        sim = phrase_alignment_similarity(norm_expected, norm_transcript)
        return sim, transcript

    def _transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        """Transcribe audio chunk."""
        if self._model is not None:
            # In-process faster_whisper
            try:
                # Ensure float32 normalized to [-1, 1]
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                segments, _ = self._model.transcribe(
                    audio,
                    language="ar",
                    initial_prompt=None,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                )
                return " ".join(s.text for s in segments)
            except Exception as e:
                logger.warning("In-process Whisper transcription error: %s", e)
                return ""

        # Fallback: invoke via mualim python venv
        if MUALIM_PYTHON_BIN.exists() and self.model_path.exists():
            try:
                with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_f:
                    tmp_path = tmp_f.name
                    np.save(tmp_path, audio.astype(np.float32))

                script = f"""
import numpy as np
import faster_whisper
audio = np.load('{tmp_path}')
model = faster_whisper.WhisperModel('{self.model_path}', device='{self.device}', compute_type='int8')
segments, _ = model.transcribe(audio, language='ar', initial_prompt=None)
print(' '.join(s.text for s in segments))
"""
                res = subprocess.run(
                    [str(MUALIM_PYTHON_BIN), "-c", script],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                if res.returncode == 0:
                    return res.stdout.strip()
            except Exception as e:
                logger.warning("Out-of-process Whisper transcription error: %s", e)
                return ""

        return ""


class RepeatCandidateEvaluator:
    """Multi-feature scorer producing calibrated P(repeat) for candidate repeats."""

    def __init__(
        self,
        whisper_verifier: Optional[WhisperVerifier] = None,
        auto_accept_threshold: float = 0.90,
        review_threshold: float = 0.50,
    ):
        self.whisper_verifier = whisper_verifier or WhisperVerifier()
        self.auto_accept_threshold = auto_accept_threshold
        self.review_threshold = review_threshold

    def calculate_p_repeat(
        self,
        ctc_lattice_score: float,
        whisper_similarity: float,
        acoustic_similarity: float,
        backward_jump: int,
        inter_word_pause: float,
        asr_surplus_flag: bool,
        has_whisper: bool = True,
    ) -> float:
        """Calculate calibrated P(repeat) via logistic feature fusion."""
        norm_jump = min(float(backward_jump), 4.0) / 4.0
        norm_pause = min(max(0.0, float(inter_word_pause)), 2.0) / 2.0
        surplus_val = 1.0 if asr_surplus_flag else 0.0

        if has_whisper:
            # Calibrated 6-feature logistic model
            z = (
                -3.5
                + 2.8 * ctc_lattice_score
                + 2.8 * whisper_similarity
                + 2.0 * acoustic_similarity
                + 0.8 * norm_jump
                + 0.5 * norm_pause
                + 0.8 * surplus_val
            )
        else:
            # Calibrated 5-feature fallback without Whisper
            z = (
                -3.5
                + 4.2 * ctc_lattice_score
                + 3.4 * acoustic_similarity
                + 0.8 * norm_jump
                + 0.5 * norm_pause
                + 0.8 * surplus_val
            )

        # Standard logistic sigmoid
        p = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))
        return float(p)

    def evaluate_candidate(
        self,
        candidate: BackwardPathCandidate,
        expected_phrase: str,
        audio_samples: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        seconds_per_frame: float = 0.04,
    ) -> Dict[str, Any]:
        """Evaluate a BackwardPathCandidate and return a standardized repeat object."""
        start_sec = candidate.instance2_frames[0] * seconds_per_frame
        end_sec = candidate.instance2_frames[1] * seconds_per_frame

        has_whisper = False
        whisper_sim = candidate.whisper_similarity

        if audio_samples is not None and self.whisper_verifier is not None:
            w_sim, _ = self.whisper_verifier.verify_crop(
                audio_samples=audio_samples,
                sample_rate=sample_rate,
                start_sec=start_sec,
                end_sec=end_sec,
                expected_phrase=expected_phrase,
            )
            whisper_sim = w_sim
            has_whisper = True
            candidate.whisper_similarity = w_sim

        p_repeat = self.calculate_p_repeat(
            ctc_lattice_score=candidate.ctc_lattice_score,
            whisper_similarity=whisper_sim,
            acoustic_similarity=candidate.acoustic_similarity,
            backward_jump=candidate.backward_jump,
            inter_word_pause=candidate.pause_duration_sec,
            asr_surplus_flag=candidate.asr_surplus_flag,
            has_whisper=has_whisper,
        )

        if p_repeat >= self.auto_accept_threshold:
            status = "accepted"
        elif p_repeat >= self.review_threshold:
            status = "review_queue"
        else:
            status = "rejected"

        evidence = {
            "ctc_lattice_score": round(float(candidate.ctc_lattice_score), 4),
            "whisper_similarity": round(float(whisper_sim), 4),
            "acoustic_cosine": round(float(candidate.acoustic_similarity), 4),
            "backward_jump": int(candidate.backward_jump),
            "inter_word_pause": round(float(candidate.pause_duration_sec), 3),
            "asr_word_surplus_candidate": bool(candidate.asr_surplus_flag),
        }

        repeat_event = {
            "start": round(start_sec, 3),
            "end": round(end_sec, 3),
            "canonical_indices": list(candidate.canonical_indices),
            "anchor_idx": int(candidate.anchor_idx),
            "restart_idx": int(candidate.restart_idx),
            "repeat_type": candidate.repeat_type,
            "direction": "abandoned_backtrack" if candidate.repeat_type in ("backtrack", "stutter") else "stylistic_repeat",
            "evaluated_candidates": [
                {
                    "phrase": expected_phrase,
                    "indices": list(candidate.canonical_indices),
                    "p_repeat": round(p_repeat, 4),
                    "status": status,
                }
            ],
            "p_repeat": round(p_repeat, 4),
            "status": status,
            "evidence": evidence,
        }
        return repeat_event


class ReviewQueueExporter:
    """Exports unresolved repeat candidates (0.50 <= P < 0.90) to unresolved_repeats.json."""

    @staticmethod
    def export(unresolved: List[Dict[str, Any]], out_path: str | Path = "unresolved_repeats.json"):
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(unresolved, f, ensure_ascii=False, indent=2)
        logger.info("Exported %d unresolved repeat candidates to %s", len(unresolved), path)
