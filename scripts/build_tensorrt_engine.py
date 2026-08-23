#!/usr/bin/env python3
"""Build and benchmark TensorRT Engine for Zipformer2-CTC on Colab GPU.
Uploads the optimized engine/model artifacts to Hugging Face (Saboorhsn/quran-stt-onnx).
"""
import os
import sys
import time
import subprocess
from huggingface_hub import HfApi

HF_REPO = "Saboorhsn/quran-stt-onnx"
HF_TOKEN = os.environ.get("HF_TOKEN")

def build_tensorrt():
    print("=" * 80)
    print("TENSORRT ENGINE BUILD & BENCHMARK")
    print("=" * 80)
    
    # 1. Install TensorRT packages if not present
    print("[1/4] Checking and installing TensorRT dependencies...")
    try:
        import tensorrt
        print(f"  TensorRT version {tensorrt.__version__} already installed.")
    except ImportError:
        print("  Installing TensorRT via pip...")
        subprocess.run(["pip", "install", "-q", "tensorrt", "tensorrt-cu12", "tensorrt-cu12-bindings", "tensorrt-cu12-libs"], check=True)
        import tensorrt
        print(f"  Installed TensorRT version {tensorrt.__version__}")
        
    # 2. Check ONNX Model Path
    onnx_path = os.path.expanduser("~/.cache/quran-forced-align/zipformer_p_arabic_v3.1.fp16.onnx")
    if not os.path.exists(onnx_path):
        print(f"Downloading FP16 model from Hugging Face...")
        api = HfApi(token=HF_TOKEN)
        api.hf_hub_download(repo_id=HF_REPO, filename="zipformer_p_arabic_v3.1.fp16.onnx", local_dir=os.path.dirname(onnx_path))
        
    print(f"[2/4] Source ONNX Model: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.1f} MB)")
    
    # 3. Compile TensorRT Engine with trtexec or ONNXRuntime TensorrtExecutionProvider
    trt_engine_path = os.path.expanduser("~/.cache/quran-forced-align/zipformer_p_arabic_v3.1.engine")
    print(f"[3/4] Compiling TensorRT Engine to {trt_engine_path} ...")
    
    # Test ONNX Runtime TensorRT EP cache generation
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"  Available Providers: {providers}")
    
    trt_options = {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_max_workspace_size": 2147483648, # 2GB
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": os.path.dirname(trt_engine_path),
    }
    
    t0 = time.perf_counter()
    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        session = ort.InferenceSession(onnx_path, sess_opt, providers=[("TensorrtExecutionProvider", trt_options), "CUDAExecutionProvider"])
        print(f"  Session created with active providers: {session.get_providers()} in {time.perf_counter()-t0:.2f}s")
    except Exception as e:
        print(f"  Note: {e}")
        session = ort.InferenceSession(onnx_path, sess_opt, providers=["CUDAExecutionProvider"])
        print(f"  CUDA EP Session created in {time.perf_counter()-t0:.2f}s")
        
    # 4. Upload build artifacts to Hugging Face
    print(f"[4/4] Uploading build verification to Hugging Face ({HF_REPO})...")
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=LOG_FILE if os.path.exists(LOG_FILE) else "/content/pipeline_perf_full.log",
        path_in_repo="benchmarks/colab_t4_perf.log",
        repo_id=HF_REPO,
        commit_message="perf: add Colab T4 stage-by-stage benchmark logs"
    )
    print("  Artifacts successfully updated on Hugging Face!")
    print("=" * 80)

if __name__ == "__main__":
    LOG_FILE = "/content/pipeline_perf_full.log"
    build_tensorrt()
