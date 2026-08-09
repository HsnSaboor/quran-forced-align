"""Word-level cue extraction + repeat detection/local re-alignment.

Adapted from build_surah_srt.py's two-pass repeat-detection design, but
implemented as a forced-alignment re-run (against a DOUBLED local reference)
rather than a second banded-edit-distance pass.
"""
from .detection import detect_and_fix_repeats
from .spans import extract_word_frame_spans, token_frame_spans

__all__ = [
    "token_frame_spans",
    "extract_word_frame_spans",
    "detect_and_fix_repeats",
]
