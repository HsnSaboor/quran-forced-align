#!/usr/bin/env python3
import os
import sys
import time
import json
import torch
import numpy as np

# Read HF_TOKEN from env or argument
hf_token = os.environ.get("HF_TOKEN")

sys.path.insert(0, "/content/quran-forced-align/src")

from quran_forced_align.pipeline import align_surah
from quran_forced_align.model_manager import resolve_model, resolve_tokens

def main():
    print("=" * 80, flush=True)
    print("SURAH 2 FULL END-TO-END PERFORMANCE BENCHMARK & STAGE PROFILING", flush=True)
    print("=" * 80, flush=True)
    
    mp3_path = "/content/verification_output/abdel-mohsen-al-obeikan_002.mp3"
    if not os.path.exists(mp3_path):
        import urllib.request
        os.makedirs(os.path.dirname(mp3_path), exist_ok=True)
        url = "https://media.assabile.com/assabile/recitations_7892537823/mp3/abdel-mohsen-al-obeikan/abdalmhsn-al-bykan-002-al-baqara-234-2232.mp3"
        print(f"Downloading test audio: {url} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(mp3_path, "wb") as f:
            f.write(resp.read())
        print(f"Downloaded {os.path.getsize(mp3_path)/(1024*1024):.1f} MB", flush=True)
        
    model_path = resolve_model(device="cuda", prefer_fp16=True)
    tokens_tuple = resolve_tokens()
    
    print(f"Audio file:  {mp3_path}", flush=True)
    print(f"Model path:  {model_path}", flush=True)
    print(f"Tokens path: {tokens_tuple}", flush=True)
    print(f"GPU Device:  {torch.cuda.get_device_name(0)}", flush=True)
    
    # GPU warm-up
    torch.cuda.init()
    _ = torch.zeros((10, 10), device="cuda")
    torch.cuda.synchronize()
    
    print("\n" + "=" * 80, flush=True)
    print("RUNNING END-TO-END BENCHMARK WITH DETAILED LOGGING", flush=True)
    print("=" * 80, flush=True)
    
    t_start = time.perf_counter()
    result = align_surah(
        surah=2,
        audio_path=mp3_path,
        model_path=model_path,
        tokens_path=tokens_tuple,
        device="cuda",
        intra_surah_split=True,
        verbose=True,
    )
    t_end = time.perf_counter() - t_start
    audio_dur = result[-1]["end"]
    
    print("\n" + "=" * 80, flush=True)
    print(f"FINAL BENCHMARK RESULT:")
    print(f"  Audio Duration:  {audio_dur:.2f}s ({audio_dur/60:.1f} minutes)")
    print(f"  Wall-Clock Time: {t_end:.3f}s")
    print(f"  Overall Speed:   {audio_dur / max(0.001, t_end):.1f}x realtime")
    print(f"  Total Cues:      {len(result)}")
    print(f"  Repeats Found:   {len([w for w in result if w.get('is_repeat')])}")
    print(f"  Low Conf Rate:   {100*len([w for w in result if w.get('low_confidence')])/len(result):.2f}%")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
