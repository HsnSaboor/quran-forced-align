#!/usr/bin/env python3
"""Full-file Surah 2 alignment and acoustic repeat verification on Colab
for multiple reciters from verified_curl_cffi.json without opus transcoding.
"""
import os
import sys
import json
import time
import urllib.request
import numpy as np

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

def verify_reciter(slug, mp3_url, model_path, tokens_tuple, out_dir):
    print("\n" + "=" * 80, flush=True)
    print(f"VERIFYING RECITER: {slug} (Surah 2 Al-Baqarah)", flush=True)
    print("=" * 80, flush=True)
    
    mp3_path = os.path.join(out_dir, f"{slug}_002.mp3")
    json_path = os.path.join(out_dir, f"{slug}_002_aligned.json")
    
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
        
    total_words = len(result)
    repeat_words = [w for w in result if w.get("is_repeat")]
    low_conf_words = [w for w in result if w.get("low_confidence")]
    
    print("-" * 80, flush=True)
    print(f"SUMMARY FOR {slug}:", flush=True)
    print(f"  Total Aligned Cues: {total_words}", flush=True)
    print(f"  Repeated Words:     {len(repeat_words)}", flush=True)
    print(f"  Low Conf Words:     {len(low_conf_words)} ({100*len(low_conf_words)/max(1,total_words):.2f}%)", flush=True)
    print(f"  Processing Time:    {elapsed:.2f}s ({len(result)} words in {elapsed:.1f}s)", flush=True)
    
    if repeat_words:
        print("\n  Sample Detected Repeats:", flush=True)
        for r in repeat_words[:12]:
            w = r.get("word")
            a = r.get("aya")
            s = r.get("start")
            e = r.get("end")
            alp = r.get("avg_logprob")
            alp_str = f"{alp:.3f}" if alp is not None else "N/A"
            print(f"    Ayah {a:3d} | {s:7.2f}s -> {e:7.2f}s | {w:16s} | logp: {alp_str}", flush=True)
            
    for target_aya in (91, 109):
        target_reps = [w for w in repeat_words if w.get("aya") == target_aya]
        if target_reps:
            print(f"\n  Ayah {target_aya} Repeats ({len(target_reps)} words):", flush=True)
            for tr in target_reps:
                w = tr.get("word")
                s = tr.get("start")
                e = tr.get("end")
                alp = tr.get("avg_logprob")
                alp_str = f"{alp:.3f}" if alp is not None else "N/A"
                print(f"    {s:7.2f}s -> {e:7.2f}s | {w:16s} | logp: {alp_str}", flush=True)
        else:
            print(f"  Ayah {target_aya}: No repeats in this recitation.", flush=True)
            
    print("=" * 80, flush=True)
    return {
        "slug": slug,
        "total_words": total_words,
        "repeats": len(repeat_words),
        "low_conf": len(low_conf_words),
        "time_sec": elapsed,
        "json_path": json_path,
    }

def main():
    repo_dir = "/content/quran-forced-align"
    json_catalog = os.path.join(repo_dir, "verified_curl_cffi.json")
    out_dir = "/content/verification_output"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Auto-resolving model and tokens...", flush=True)
    model_path = resolve_model(device="cuda", prefer_fp16=True)
    tokens_path = resolve_tokens()
    print(f"Model path: {model_path}", flush=True)
    print(f"Tokens path: {tokens_path}", flush=True)
    
    tok2id, id2tok, blank_id, max_token_len = load_tokens(tokens_path)
    tokens_tuple = (tok2id, id2tok, blank_id, max_token_len)
    
    with open(json_catalog, "r", encoding="utf-8") as f:
        catalog = json.load(f)
        
    reciters = catalog.get("verified", [])
    print(f"Total available reciters in catalog: {len(reciters)}", flush=True)
    
    # Selection of reciters for Surah 2 verification
    chosen_slugs = [
        "abdallah-kamel",
        "abdel-mohsen-al-obeikan",
        "abdelali-anoun",
        "abdelmoujib-benkirane",
        "abdul-aziz-al-ahmed"
    ]
    
    results = []
    for r in reciters:
        slug = r.get("slug")
        if slug in chosen_slugs:
            s2 = r.get("surahs", {}).get("2")
            if s2 and s2.get("http_200"):
                url = s2.get("mp3_url")
                try:
                    res = verify_reciter(slug, url, model_path, tokens_tuple, out_dir)
                    results.append(res)
                except Exception as e:
                    print(f"Error verifying {slug}: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                
    print("\n" + "#" * 80, flush=True)
    print("FINAL MULTI-RECITER VERIFICATION SUMMARY (SURAH 2):", flush=True)
    print("#" * 80, flush=True)
    print(f"{'Reciter Slug':<28} | {'Words':<7} | {'Repeats':<8} | {'Low Conf':<9} | {'Time (s)':<8}", flush=True)
    print("-" * 75, flush=True)
    for res in results:
        print(f"{res['slug']:<28} | {res['total_words']:<7} | {res['repeats']:<8} | {res['low_conf']:<9} | {res['time_sec']:<8.1f}", flush=True)
    print("#" * 80, flush=True)

if __name__ == "__main__":
    main()
