"""Verifies that feeding a numpy float32 array directly into
kaldi_native_fbank's accept_waveform produces byte-identical fbank output
compared to feeding a Python list (via `.tolist()`) of the same samples.

This underpins the optimization in features.py's compute_fbank_features,
which passes the numpy array directly (skipping the list-boxing pass) on
the assumption that this is a transport-mechanism change only, not a
numeric one. Does not require the ONNX model, so it is not skipped by
conftest's model-presence check.
"""
import kaldi_native_fbank as knf
import numpy as np

from quran_forced_align.constants import SAMPLE_RATE


def _make_opts():
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = SAMPLE_RATE
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.snip_edges = True
    opts.frame_opts.dither = 0
    opts.mel_opts.num_bins = 80
    opts.use_energy = False
    return opts


def _run(samples):
    fb = knf.OnlineFbank(_make_opts())
    fb.accept_waveform(SAMPLE_RATE, samples)
    fb.input_finished()
    n_frames = fb.num_frames_ready
    feats = np.empty((n_frames, 80), dtype=np.float32)
    for i in range(n_frames):
        feats[i] = fb.get_frame(i)
    return feats


def test_array_input_byte_identical_to_list_input():
    rng = np.random.default_rng(0)
    samples = rng.uniform(-1.0, 1.0, size=SAMPLE_RATE).astype(np.float32)

    feats_from_array = _run(samples)
    feats_from_list = _run(samples.tolist())

    assert feats_from_array.shape == feats_from_list.shape
    assert feats_from_array.tobytes() == feats_from_list.tobytes()


def test_array_input_byte_identical_on_short_and_silent_signals():
    for samples in (
        np.zeros(4000, dtype=np.float32),
        np.linspace(-0.5, 0.5, 1600, dtype=np.float32),
    ):
        feats_from_array = _run(samples)
        feats_from_list = _run(samples.tolist())
        assert feats_from_array.tobytes() == feats_from_list.tobytes()
