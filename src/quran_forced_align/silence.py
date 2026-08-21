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


def find_silence_midpoints(samples, sample_rate=16000, max_splits=None, min_gap_sec=10.0, target_segment_sec=45.0):
    """Return a sorted list of sample-index positions for optimal intra-surah parallelism.
    
    Uses vectorized moving-average RMS energy computation. If the audio is longer than
    `target_segment_sec`, partitions the recording into balanced segments of ~target_segment_sec
    (e.g. 45s) by locating the local energy minimum within a search window around each boundary.
    This guarantees equal stream lengths across all parallel GPU streams, eliminating stragglers
    and maximizing Tensor Core occupancy.
    """
    if sample_rate != 16000:
        raise ValueError(
            f"find_silence_midpoints: expected 16kHz samples, got sample_rate={sample_rate}"
        )
    n_samples = len(samples)
    if n_samples < _FRAME_LEN_SAMPLES:
        return []

    frame_len = _FRAME_LEN_SAMPLES
    hop = _HOP_SAMPLES
    n_frames = (n_samples - frame_len) // hop + 1

    # High-speed vectorized moving-average energy computation.
    # For very long audio (>10 min), downsample first to avoid
    # computing prefix sums over 100M+ elements.
    if n_frames > 100000:
        # Downsample: compute RMS energy over every hop-th block
        # by reshaping into blocks and taking mean of squares
        block_size = frame_len
        n_blocks = n_samples // block_size
        if n_blocks > 0:
            reshaped = samples[:n_blocks * block_size].reshape(n_blocks, block_size).astype(np.float32)
            block_energy = np.sqrt(np.mean(reshaped ** 2, axis=1) + 1e-12)
            # Map frame indices to block indices
            frame_to_block = (np.arange(n_frames) * hop) // block_size
            frame_to_block = np.clip(frame_to_block, 0, n_blocks - 1)
            energies = block_energy[frame_to_block]
        else:
            energies = np.ones(n_frames, dtype=np.float32)
    else:
        squared = samples.astype(np.float32) ** 2
        cum = np.pad(np.cumsum(squared, dtype=np.float64), (1, 0))
        frame_indices = np.arange(n_frames) * hop
        energies = np.sqrt(np.maximum(0, (cum[frame_indices + frame_len] - cum[frame_indices]) / frame_len) + 1e-12)

    total_sec = n_samples / sample_rate

    # If audio is long enough, use balanced target-interval splitting
    if total_sec > (target_segment_sec * 1.2):
        n_segments = int(round(total_sec / target_segment_sec))
        if max_splits is not None:
            n_segments = min(n_segments, max_splits + 1)
        
        target_samples = [int(i * target_segment_sec * sample_rate) for i in range(1, n_segments)]
        search_window_samples = int(min(target_segment_sec * 0.25, 6.0) * sample_rate)
        splits = []
        for target in target_samples:
            s_start = max(0, target - search_window_samples)
            s_end = min(n_samples - frame_len, target + search_window_samples)
            f_start = s_start // hop
            f_end = s_end // hop
            if f_end > f_start:
                min_f = f_start + int(np.argmin(energies[f_start:f_end]))
                splits.append(int(min_f * hop + frame_len // 2))
        return sorted(splits)

    # Standard silence run detection for shorter recordings
    threshold = np.percentile(energies, _SILENCE_PERCENTILE)
    is_silence = energies <= threshold
    min_run_frames = int(round(_MIN_SILENCE_RUN_SEC * 16000 / hop))
    
    runs = []
    i = 0
    while i < n_frames:
        if is_silence[i]:
            j = i
            while j < n_frames and is_silence[j]:
                j += 1
            if j - i >= min_run_frames:
                mid_frame = (i + j) // 2
                midpoint_sample = mid_frame * hop + frame_len // 2
                runs.append((midpoint_sample, j - i))
            i = j
        else:
            i += 1

    if not runs:
        return []

    if max_splits is not None and len(runs) > max_splits:
        runs_by_len = sorted(runs, key=lambda r: r[1], reverse=True)
        min_gap_samples = int(min_gap_sec * 16000)
        selected = []
        for pos, run_len in runs_by_len:
            if all(abs(pos - chosen) >= min_gap_samples for chosen in selected):
                selected.append(pos)
                if len(selected) >= max_splits:
                    break
        return sorted(selected)

    return sorted(r[0] for r in runs)
