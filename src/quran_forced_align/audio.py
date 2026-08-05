"""Generic ffmpeg-based audio loader. Ported from build_surah_srt.py (the
older, superseded script) -- this one function's logic is a generic, reused
utility with nothing to do with that script's own (buggy) alignment
pipeline.
"""
import os
import subprocess
import tempfile
import wave

import numpy as np

from .constants import SAMPLE_RATE


def load_audio_as_wav16k(path):
    """Convert any input audio (mp3/etc) to 16kHz mono PCM via ffmpeg,
    return float32 samples in [-1,1]."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", tmp_path],
        check=True, capture_output=True,
    )
    with wave.open(tmp_path, "rb") as wf:
        n = wf.getnframes()
        data = wf.readframes(n)
    os.unlink(tmp_path)
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
