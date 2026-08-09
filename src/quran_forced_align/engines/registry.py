"""Engine name -> constructor lookup, shared by `cli.py`/`batch_cli.py`'s
`--device` flag so both entry points parse and validate the same set of
engine names identically.

`cuda.py` is imported lazily (only when `"cuda"` is actually requested),
not at this module's import time -- it depends on torch/torchaudio/
onnxruntime-gpu (see pyproject.toml's `cuda` extra), which a CPU-only
install (this package's default dependency set) never installs. Every
caller that only ever requests `"cpu"` -- which is every existing caller
before this feature existed -- must keep working with zero torch-related
import errors, exactly as before.
"""
from .cpu import CPUEngine

_ENGINE_NAMES = ("cpu", "cuda")


def get_engine(name):
    """Return the `Engine` constructor for `name` (`"cpu"` or `"cuda"`).
    Raises `ValueError` with the full list of valid names on an
    unrecognized one, so `argparse`'s `choices=` validation and this
    lookup give the user the same clear error either way."""
    if name == "cpu":
        return CPUEngine
    if name == "cuda":
        from .cuda import CUDAEngine
        return CUDAEngine
    raise ValueError(f"unknown forced-alignment engine {name!r} -- valid choices: {list(_ENGINE_NAMES)}")
