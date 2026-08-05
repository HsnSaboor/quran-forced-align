"""Generic ffmpeg-based audio loader. Ported from build_surah_srt.py (the
older, superseded script) -- this one function's logic is a generic, reused
utility with nothing to do with that script's own (buggy) alignment
pipeline.
"""
import subprocess

import numpy as np

from .constants import SAMPLE_RATE


def load_audio_as_wav16k(path):
    """Convert any input audio (mp3/etc) to 16kHz mono PCM via ffmpeg,
    return float32 samples in [-1,1].

    Streams raw PCM straight from ffmpeg's stdout instead of writing a
    temporary WAV file and reading it back -- for a long recording (e.g.
    Al-Baqarah at ~117 minutes, ~225MB of 16kHz mono s16 PCM) the previous
    approach paid a full disk write + a full disk read for data that's
    only ever used once, in memory, immediately after. `-f s16le` (raw
    little-endian s16 samples, no WAV container) needs no header parsing
    on the read side, and is byte-for-byte the same sample values ffmpeg's
    WAV writer would have encoded for identical `-ar`/`-ac` decode
    parameters -- confirmed empirically (same decoder path, same resample
    filter, only the output container/transport differs).
    """
    proc = subprocess.run(
        ["ffmpeg", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
