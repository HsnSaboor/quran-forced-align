"""Fast C++ / ctypes wrapper for chunked parallel Kaldi fbank feature extraction."""
import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import numpy as np

_LIB_INSTANCE = None
_INIT_ATTEMPTED = False


def _get_cache_dir() -> str:
    cache_base = os.environ.get("XDG_CACHE_HOME")
    if not cache_base:
        cache_base = os.path.expanduser("~/.cache")
    d = os.path.join(cache_base, "quran_forced_align")
    os.makedirs(d, exist_ok=True)
    return d


def _find_compiler() -> str | None:
    for comp in ("g++", "clang++", "c++"):
        p = shutil.which(comp)
        if p:
            return p
    return None


def _compile_and_load():
    global _LIB_INSTANCE
    import kaldi_native_fbank as knf

    knf_dir = os.path.dirname(knf.__file__)
    inc_dir = os.path.join(knf_dir, "include")
    lib_dir = os.path.join(knf_dir, "lib")
    core_lib = os.path.join(lib_dir, "libkaldi-native-fbank-core.so")

    if not os.path.isdir(inc_dir) or not os.path.exists(core_lib):
        return None

    cpp_path = os.path.join(os.path.dirname(__file__), "_fast_fbank.cpp")
    if not os.path.isfile(cpp_path):
        return None

    with open(cpp_path, "rb") as f:
        src_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    cache_dir = _get_cache_dir()
    so_name = f"libfast_fbank_{src_hash}_{sys.platform}_{sys.version_info.major}{sys.version_info.minor}.so"
    so_path = os.path.join(cache_dir, so_name)

    if not os.path.exists(so_path):
        compiler = _find_compiler()
        if not compiler:
            return None

        # Try compiling with OpenMP first, fall back to non-OpenMP
        flags_base = [
            "-O3", "-shared", "-fPIC", "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
            cpp_path,
            f"-I{inc_dir}",
            f"-L{lib_dir}",
            "-lkaldi-native-fbank-core",
            f"-Wl,-rpath,{lib_dir}",
            "-o", so_path,
        ]

        compiled = False
        for omp_flag in (["-fopenmp"], []):
            cmd = [compiler] + omp_flag + flags_base
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=True)
                compiled = True
                break
            except Exception:
                continue

        if not compiled or not os.path.exists(so_path):
            return None

    try:
        lib = ctypes.CDLL(so_path)
        lib.compute_fbank_c.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.compute_fbank_c.restype = ctypes.c_int
        return lib
    except Exception:
        return None


def get_fast_fbank_lib():
    global _LIB_INSTANCE, _INIT_ATTEMPTED
    if not _INIT_ATTEMPTED:
        _INIT_ATTEMPTED = True
        try:
            _LIB_INSTANCE = _compile_and_load()
        except Exception:
            _LIB_INSTANCE = None
    return _LIB_INSTANCE


def fast_compute_fbank(samples: np.ndarray, tail_silence_sec: float = 0.3) -> np.ndarray | None:
    lib = get_fast_fbank_lib()
    if lib is None:
        return None

    samples = np.ascontiguousarray(samples, dtype=np.float32)
    tail_samples = int(16000 * tail_silence_sec)
    total_samples = len(samples) + tail_samples
    frame_length = 400
    frame_shift = 160

    if total_samples < frame_length:
        return np.empty((0, 80), dtype=np.float32)

    n_frames = (total_samples - frame_length) // frame_shift + 1
    feats = np.empty((n_frames, 80), dtype=np.float32)

    p_in = samples.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    p_out = feats.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    actual_frames = lib.compute_fbank_c(p_in, len(samples), tail_samples, p_out)
    if actual_frames != n_frames:
        return None
    return feats
