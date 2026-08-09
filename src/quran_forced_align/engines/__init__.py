"""Pluggable forced-alignment execution engines.

Two engines implement the SAME contract (see `base.Engine`): given a
surah's fbank features and its combined reference token ids, produce the
full per-frame log-probability matrix (`run_inference`) and the
blank-interleaved forced-alignment result (`forced_align`) -- the
`(ext, path, margins)` triple every downstream consumer (`repeats.py`,
`confidence.py`, `srt.py`) already depends on, unchanged regardless of
which engine produced it.

  - `cpu`: the original raw-ONNX CPUExecutionProvider + hand-rolled numpy
    Viterbi (see `onnx_model.py`/`viterbi.py`) -- unchanged, still the
    default, still bit-identical to every previous release.
  - `cuda`: onnxruntime's CUDAExecutionProvider for inference +
    `torchaudio.functional.forced_align` (a compiled CUDA kernel) for
    alignment -- for batch-processing many surahs/reciters where GPU
    throughput actually helps (see `cuda.py`'s module docstring for the
    full rationale and the numerics this was validated against on a real
    T4 GPU).

`get_engine(name)` is the only public entry point most callers need.
"""
from .base import Engine
from .registry import get_engine

__all__ = ["Engine", "get_engine"]
