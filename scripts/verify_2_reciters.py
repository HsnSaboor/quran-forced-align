#!/usr/bin/env python3
import os
import sys
import json
import time
import urllib.request

sys.path.insert(0, "/content/quran-forced-align/src")

from quran_forced_align.pipeline import align_surah
from quran_forced_align.tokenizer import load_tokens
from quran_forced_align.model_manager import resolve_model, resolve_tokens

def download_file(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        print(f"  [Cache hit] {out_path} ({os.path.getsize(out_path)/(1024*1024):.1f} MB)")
        return out_path
    print(f"  Downloading: {url} -> {out_path} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        f.write(resp.read())
    print(f"  Downloaded: {os.path.getsize(out_path)/(1024*1024):.1f} MB", flush=True)
    return out_path

def main():
    with open("/content/quran-forced-align/verified_curl_cffi.json", "r") as f:
        catalog = json.load(f)

    catalog_map = {item["slug"]: item for item in catalog["verified"]}
    target_reciters = ["abdallah-kamel", "abdel-mohsen-al-obeikan"]
    
    out_dir = "/content/verification_output"
    os.makedirs(out_dir, exist_ok=True)
    
    model_path = resolve_model("zipformer_p_arabic_v3.1.fp16.onnx")
    tokens_tuple = resolve_tokens("/content/quran-forced-align/model/tokens.txt")
    
    for slug in target_reciters:
        info = catalog_map[slug]
        mp3_url = info["surahs"]["2"]["mp3_url"]
        mp3_path = os.path.join(out_dir, f"{slug}_002.mp3")
        json_path = os.path.join(out_dir, f"{slug}_002_aligned.json")
        
        print(f"\n==================================================", flush=True)
        print(f"Aligning {slug} (Surah 2)...", flush=True)
        print(f"==================================================", flush=True)
        download_file(mp3_url, mp3_path)
        
        t0 = time.perf_counter()
        result = align_surah(
            surah=2,
            audio_path=mp3_path,
            model_path=model_path,
            tokens_path=tokens_tuple,
            device="cuda",
            include_istiaatha="auto",
            verbose=True,
        )
        elapsed = time.perf_counter() - t0
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        print(f"Done {slug}: {len(result)} cues saved to {json_path} in {elapsed:.1f}s", flush=True)

if __name__ == "__main__":
    main()
