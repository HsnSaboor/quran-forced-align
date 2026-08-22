"""Centralized Model, Token, and Device Manager for Quran Forced Alignment.

Handles auto-discovery, Hugging Face caching, and automatic device detection.
Zero configuration required on local machines, Colab notebooks, or production servers.
"""
import os
import sys
import urllib.request
from pathlib import Path
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

HF_REPO = "Saboorhsn/quran-stt-onnx"
DEFAULT_FP16_MODEL = "zipformer_p_arabic_v3.1.fp16.onnx"
DEFAULT_FP32_MODEL = "zipformer_p_arabic_v3.1.onnx"
DEFAULT_INT8_MODEL = "zipformer_p_arabic_v3.1.int8.onnx"
DEFAULT_TOKENS = "tokens.txt"

CACHE_DIR = Path(os.environ.get("QURAN_FORCED_ALIGN_CACHE_DIR", Path.home() / ".cache" / "quran-forced-align"))


def verify_colab_environment() -> Dict[str, Any]:
    """Verify runtime environment, GPU availability, and library sanity."""
    is_colab = "google.colab" in sys.modules or os.path.exists("/content")
    gpu_available = False
    gpu_name = None
    cuda_version = None
    ort_gpu_ok = False
    ffmpeg_ok = shutil.which("ffmpeg") is not None

    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
    except Exception:
        pass

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        ort_gpu_ok = "CUDAExecutionProvider" in providers
    except Exception:
        pass

    status = {
        "is_colab": is_colab,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "ort_gpu_ok": ort_gpu_ok,
        "ffmpeg_installed": ffmpeg_ok,
    }

    if is_colab and gpu_available and not ort_gpu_ok:
        print("⚠️ [Colab Warning] GPU detected but onnxruntime-gpu is not linked properly.")
        print("   Fix: Run `pip uninstall -y onnxruntime && pip install onnxruntime-gpu==1.20.2`")

    if not ffmpeg_ok:
        print("⚠️ [Warning] ffmpeg not found on PATH. Audio decoding will fail.")

    return status


def get_hf_url(filename: str) -> str:
    return f"https://huggingface.co/{HF_REPO}/resolve/main/{filename}"


def _download_file(url: str, target_path: Path, desc: str = "asset") -> Path:
    """Download a file with progress reporting and Hugging Face auth support."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    token = os.environ.get("HF_TOKEN")

    print(f"Downloading {desc} from Hugging Face ({HF_REPO})...")
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req) as resp, open(temp_path, "wb") as out_file:
            total_size = int(resp.headers.get("content-length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1MB blocks

            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r  [{downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB] ({pct:.1f}%)", end="", flush=True)

            print(f"\n  Saved to: {target_path}")
        temp_path.replace(target_path)
        return target_path
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {desc} from {url}: {e}") from e


def resolve_device(device_arg: str = "auto") -> str:
    """Resolve compute device ('auto' -> 'cuda' if GPU execution provider available, else 'cpu')."""
    if device_arg and device_arg.lower() != "auto":
        return device_arg.lower()

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            import torch
            if torch.cuda.is_available():
                return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_tokens(tokens_path: Optional[str] = None) -> str:
    """Find or auto-download tokens.txt."""
    if tokens_path and os.path.exists(tokens_path):
        return tokens_path

    search_paths = [
        Path("model/tokens.txt"),
        Path(__file__).parent.parent.parent / "model" / "tokens.txt",
        CACHE_DIR / "tokens.txt",
    ]

    for p in search_paths:
        if p.exists() and p.stat().st_size > 100:
            return str(p)

    # Download to cache
    target = CACHE_DIR / "tokens.txt"
    _download_file(get_hf_url(DEFAULT_TOKENS), target, desc="tokens.txt")
    return str(target)


def resolve_model(
    model_path: Optional[str] = None,
    device: str = "auto",
    prefer_fp16: bool = True
) -> str:
    """Find or auto-download the optimal Zipformer ONNX model."""
    if model_path and os.path.exists(model_path) and os.path.getsize(model_path) > 10000000:
        return model_path

    resolved_device = resolve_device(device)
    
    # Priority for CUDA: FP16 (if prefer_fp16) -> FP32 -> INT8
    # Priority for CPU: INT8 -> FP32
    if resolved_device == "cuda":
        candidate_filenames = [DEFAULT_FP16_MODEL, DEFAULT_FP32_MODEL, DEFAULT_INT8_MODEL] if prefer_fp16 else [DEFAULT_FP32_MODEL, DEFAULT_INT8_MODEL]
    else:
        candidate_filenames = [DEFAULT_INT8_MODEL, DEFAULT_FP32_MODEL]

    for filename in candidate_filenames:
        search_paths = [
            Path(f"model/{filename}"),
            Path(__file__).parent.parent.parent / "model" / filename,
            CACHE_DIR / filename,
        ]
        for p in search_paths:
            if p.exists() and p.stat().st_size > 10000000:
                return str(p)

    # Auto-download best candidate to cache
    downloadable_candidates = (
        [DEFAULT_FP16_MODEL, DEFAULT_FP32_MODEL, DEFAULT_INT8_MODEL]
        if (resolved_device == "cuda" and prefer_fp16)
        else [DEFAULT_INT8_MODEL, DEFAULT_FP32_MODEL]
    )
    chosen = downloadable_candidates[0]
    target = CACHE_DIR / chosen
    _download_file(get_hf_url(chosen), target, desc=f"Zipformer model ({chosen})")
    return str(target)
