"""Unit tests for engines.registry.get_engine's name->constructor lookup
-- previously untested (test_ground_truth_recall.py imports CPUEngine
directly, bypassing the registry entirely; found as a coverage gap in
code review).
"""
import pytest

from quran_forced_align.engines import get_engine
from quran_forced_align.engines.cpu import CPUEngine


def test_get_engine_cpu_returns_cpu_engine_class():
    assert get_engine("cpu") is CPUEngine


def test_get_engine_cuda_lazily_returns_cuda_engine_class():
    # Only imports quran_forced_align.engines.cuda (which itself only
    # needs onnxruntime at module scope, not torch/torchaudio -- those are
    # imported lazily inside CUDAEngine.__init__) -- must not raise even
    # without torch/torchaudio installed, matching this package's
    # documented "CPU-only installs never need torch" guarantee.
    from quran_forced_align.engines.cuda import CUDAEngine
    assert get_engine("cuda") is CUDAEngine


def test_get_engine_unknown_name_raises_value_error_listing_choices():
    with pytest.raises(ValueError, match=r"unknown forced-alignment engine 'tpu'"):
        get_engine("tpu")
