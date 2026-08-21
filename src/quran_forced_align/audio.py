"""Generic ffmpeg-based audio loader. Ported from build_surah_srt.py (the
older, superseded script) -- this one function's logic is a generic, reused
utility with nothing to do with that script's own (buggy) alignment
pipeline.
"""
import os
import subprocess

import numpy as np

from .constants import SAMPLE_RATE

AUDIO_EXTENSIONS = (".mp3", ".opus", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma")


def find_audio_file(audio_dir: str, surah: int) -> str | None:
    """Find the audio file for a given surah in audio_dir.
    
    Checks standard zero-padded formatting (e.g. '001.mp3', '001.opus')
    as well as unpadded ('1.mp3'), underscore/hyphen prefixed, or any matching extension.
    """
    if not os.path.isdir(audio_dir):
        return None

    # Check zero-padded 3-digit first (standard)
    padded = f"{surah:03d}"
    for ext in AUDIO_EXTENSIONS:
        candidate = os.path.join(audio_dir, f"{padded}{ext}")
        if os.path.isfile(candidate):
            return candidate

    # Check unpadded
    unpadded = str(surah)
    for ext in AUDIO_EXTENSIONS:
        candidate = os.path.join(audio_dir, f"{unpadded}{ext}")
        if os.path.isfile(candidate):
            return candidate

    # Scan directory for files starting with surah number
    try:
        entries = os.listdir(audio_dir)
    except OSError:
        return None

    prefix_padded = f"{padded}_"
    prefix_padded_dash = f"{padded}-"
    for fname in entries:
        if fname.startswith(prefix_padded) or fname.startswith(prefix_padded_dash):
            candidate = os.path.join(audio_dir, fname)
            if os.path.isfile(candidate):
                return candidate

    return None


def _try_load_wav_fast(path: str) -> np.ndarray | None:
    """Fast in-memory reader for uncompressed 16kHz WAV files, avoiding
    ffmpeg subprocess and pipe overhead entirely when the input is already
    16kHz PCM.
    """
    import wave

    try:
        with wave.open(path, "rb") as wf:
            if wf.getcomptype() != "NONE":
                return None
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            if framerate != SAMPLE_RATE or n_frames == 0:
                return None

            raw = wf.readframes(n_frames)
            if sampwidth == 2:
                # 16-bit signed PCM
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) * (1.0 / 32768.0)
            elif sampwidth == 4:
                # 32-bit float or int32
                data = np.frombuffer(raw, dtype=np.float32)
            else:
                return None

            if n_channels == 1:
                return data
            elif n_channels == 2:
                data = data.reshape(-1, 2)
                return (data[:, 0] + data[:, 1]) * 0.5
            else:
                return data.reshape(-1, n_channels).mean(axis=1)
    except Exception:
        return None


def _try_load_soundfile(path: str) -> np.ndarray | None:
    """Fast-path loading via soundfile if installed and sample rate matches."""
    try:
        import soundfile as sf
        info = sf.info(path)
        if info.samplerate == SAMPLE_RATE:
            data, _ = sf.read(path, dtype="float32", always_2d=False)
            if data.ndim == 1:
                return data
            elif data.ndim == 2:
                return data.mean(axis=1)
    except Exception:
        pass
    return None


def _try_load_torchaudio(path: str) -> np.ndarray | None:
    """Fast-path loading via torchaudio if installed and sample rate matches."""
    try:
        import torchaudio
        info = torchaudio.info(path)
        if info.sample_rate == SAMPLE_RATE:
            wav, sr = torchaudio.load(path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0)
            else:
                wav = wav.squeeze(0)
            return wav.numpy().astype(np.float32)
    except Exception:
        pass
    return None


def load_audio_as_wav16k(path: str, threads: int = 0) -> np.ndarray:
    """Convert any input audio (mp3/opus/wav/m4a/etc) to 16kHz mono PCM via ffmpeg,
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

    Optimized with direct fast-paths for 16kHz WAV files, optional soundfile/torchaudio
    acceleration when available, and high-throughput multi-threaded ffmpeg pipe streaming
    with non-audio stream skipping (-vn -sn -dn).
    """
    # 1. Direct uncompressed WAV fast-path
    wav_samples = _try_load_wav_fast(path)
    if wav_samples is not None:
        return wav_samples

    # 2. Soundfile fast-path (if available)
    sf_samples = _try_load_soundfile(path)
    if sf_samples is not None:
        return sf_samples

    # 3. Torchaudio fast-path (if available)
    ta_samples = _try_load_torchaudio(path)
    if ta_samples is not None:
        return ta_samples

    # 4. Universal high-throughput ffmpeg pipe streaming
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-threads", str(threads),
        "-vn", "-sn", "-dn",
        "-i", str(path),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-f", "s16le",
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=2 * 1024 * 1024,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg audio decode failed for {path}: {err}")
    return np.frombuffer(stdout, dtype=np.int16).astype(np.float32) * (1.0 / 32768.0)



def transcode_to_opus(
    input_path: str,
    output_path: str,
    loudnorm: bool = True,
    bitrate: str = "96k",
    threads: int = 0,
) -> str:
    """Transcode input audio to Opus with optional EBU R128 loudness
    normalization and dynamic range compression.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    filters = (
        "loudnorm=I=-16:TP=-1.5:LRA=11,acompressor=threshold=-25dB:ratio=3:attack=5:release=50"
        if loudnorm else "anull"
    )
    cmd = [
        "ffmpeg", "-y", "-threads", str(threads),
        "-i", input_path,
        "-af", filters,
        "-c:a", "libopus", "-b:a", bitrate, "-vbr", "on",
        "-application", "audio", "-frame_duration", "60",
        output_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg opus transcode failed ({input_path} -> {output_path}): {err}")
    return output_path
