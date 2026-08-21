"""Energy-based silence-boundary detection for intra-surah split-point
selection (see `onnx_model.run_streaming_log_probs_intra_surah_split_cuda`'s
module docstring for the full rationale this module supports).

Deliberately NOT a VAD (voice-activity-detection) library dependency
(webrtcvad/silero-vad): this package's existing dependency footprint is
already minimal (see pyproject.toml), and a plain deterministic
RMS-energy-threshold detector is sufficient for this module's ONLY job --
finding a handful of real, unambiguous pause points in a recitation to use
as intra-surah CACHE-RESET boundaries (see `onnx_model.py`'s intra-surah
splitting), not fine-grained speech/non-speech classification. Determinism
matters here as much as anywhere else in this package: an RMS-threshold
computation over a fixed waveform is exactly reproducible, with none of a
trained VAD model's own version-pinning/inference-determinism concerns to
manage on top of everything else this package already pins down.
"""
import numpy as np

from .constants import FBANK_FRAME_SHIFT_SAMPLES

_FRAME_LEN_SAMPLES = 400  # 25ms at 16kHz, matches features.py's fbank frame length
_HOP_SAMPLES = FBANK_FRAME_SHIFT_SAMPLES  # 10ms at 16kHz, matches features.py's fbank frame
# shift -- imported from constants.py (single source of truth), not re-hardcoded here, since
# pipeline.py's silence-position-to-feature-frame-index conversion needs the SAME value and a
# previous revision independently hardcoded "160" in both places with no shared reference.
_SILENCE_PERCENTILE = 5  # bottom 5% of frame energies counts as "silence" for this detector
_MIN_SILENCE_RUN_SEC = 0.3  # shortest contiguous silence run treated as a real pause, not a
# transient dip mid-word (verified empirically against real recitation audio: genuine
# ayah-boundary/phrase pauses in tested recordings run several hundred ms or more; a
# threshold this low still requires a real, sustained quiet stretch, not a single frame)


def find_silence_midpoints(samples, sample_rate=16000, max_splits=8, min_gap_sec=60.0):
    """Return a sorted list of sample-index positions, each the MIDPOINT of
    a contiguous stretch of `samples` (16kHz mono float32, as returned by
    `audio.load_audio_as_wav16k`) whose RMS energy sits in the bottom
    `_SILENCE_PERCENTILE` of the whole recording's frame-energy
    distribution for at least `_MIN_SILENCE_RUN_SEC` seconds.

    If `max_splits` is specified and more candidate silence runs exist,
    this selects the top `max_splits` longest/deepest pauses separated by at
    least `min_gap_sec` seconds to create evenly balanced, optimal intra-surah
    batch segments (preventing excessive batch dimension overhead).
    """
    if sample_rate != 16000:
        raise ValueError(
            f"find_silence_midpoints: expected 16kHz samples (this module's frame/hop "
            f"constants are tuned for that rate), got sample_rate={sample_rate}"
        )
    n_samples = len(samples)
    if n_samples < _FRAME_LEN_SAMPLES:
        return []

    n_frames = (n_samples - _FRAME_LEN_SAMPLES) // _HOP_SAMPLES + 1
    energies = np.empty(n_frames, dtype=np.float64)
    samples64 = samples.astype(np.float64)
    for i in range(n_frames):
        start = i * _HOP_SAMPLES
        seg = samples64[start:start + _FRAME_LEN_SAMPLES]
        energies[i] = np.sqrt(np.mean(seg * seg) + 1e-12)

    threshold = np.percentile(energies, _SILENCE_PERCENTILE)
    is_silence = energies <= threshold

    min_run_frames = int(round(_MIN_SILENCE_RUN_SEC * 16000 / _HOP_SAMPLES))
    runs = []  # list of (midpoint_sample, run_length_frames)
    i = 0
    while i < n_frames:
        if is_silence[i]:
            j = i
            while j < n_frames and is_silence[j]:
                j += 1
            if j - i >= min_run_frames:
                mid_frame = (i + j) // 2
                midpoint_sample = mid_frame * _HOP_SAMPLES + _FRAME_LEN_SAMPLES // 2
                runs.append((midpoint_sample, j - i))
            i = j
        else:
            i += 1

    if not runs:
        return []

    if max_splits is not None and len(runs) > max_splits:
        # Sort runs by length descending (longest pauses first)
        runs_by_len = sorted(runs, key=lambda r: r[1], reverse=True)
        min_gap_samples = int(min_gap_sec * 16000)
        selected = []
        for pos, run_len in runs_by_len:
            if all(abs(pos - chosen) >= min_gap_samples for chosen in selected):
                selected.append(pos)
                if len(selected) >= max_splits:
                    break
        # Fallback if strict spacing gave fewer than desired splits
        if len(selected) < min(max_splits, len(runs)):
            for pos, run_len in runs_by_len:
                if pos not in selected:
                    selected.append(pos)
                    if len(selected) >= max_splits:
                        break
        return sorted(selected)

    return sorted(r[0] for r in runs)
