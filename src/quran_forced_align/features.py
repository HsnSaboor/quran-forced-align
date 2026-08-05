"""Deterministic fbank feature extraction (dither=0 -- see package
docstring / module __init__ for the full determinism rationale).
"""
import kaldi_native_fbank as knf
import numpy as np

from .constants import SAMPLE_RATE


def compute_fbank_features(samples, tail_silence_sec=0.3):
    """80-dim fbank features, deterministic (dither=0), matching the
    icefall/sherpa-onnx streaming-CTC export convention: samp_freq=16000,
    frame_shift_ms=10, frame_length_ms=25, povey window, snip_edges=True,
    num_bins=80 (NOT kaldi_native_fbank's default of 23 -- must be set
    explicitly), use_log_fbank=True, dither=0.

    A short trailing silence pad is appended to the raw waveform before
    feature extraction (0.3s, matching icefall's own
    onnx_pretrained-streaming-ctc.py reference script) so the last real
    frames of speech get full right-context instead of being clipped by
    end-of-stream.
    """
    padded = np.concatenate([
        samples,
        np.zeros(int(SAMPLE_RATE * tail_silence_sec), dtype=np.float32),
    ])
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.snip_edges = True
    opts.frame_opts.dither = 0  # CRITICAL: library default is 3e-05 (nonzero) -- would break determinism
    opts.mel_opts.num_bins = 80  # NOT the knf default of 23
    opts.use_energy = False
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(SAMPLE_RATE, padded.tolist())
    fb.input_finished()
    n_frames = fb.num_frames_ready
    feats = np.stack([fb.get_frame(i) for i in range(n_frames)]).astype(np.float32)
    return feats
