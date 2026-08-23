#!/usr/bin/env python3
"""Comprehensive Stage-by-Stage & Per-Second Pipeline Profiler for Quran Forced Alignment.
Outputs microsecond-accurate metrics and logs for every step.
"""
import os
import sys
import time
import json
import urllib.request
import numpy as np
import torch

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from quran_forced_align.model_manager import resolve_model, resolve_tokens, verify_colab_environment
from quran_forced_align.audio import load_audio_as_wav16k
from quran_forced_align.features import compute_fbank_features
from quran_forced_align.silence import find_silence_midpoints
from quran_forced_align.engines import get_engine
from quran_forced_align.reference import build_combined_reference
from quran_forced_align.tokenizer import load_tokens
from quran_forced_align.trellis import frame_spans_from_path
from quran_forced_align.repeats import detect_and_fix_repeats, extract_word_frame_spans
from quran_forced_align.confidence import flag_low_confidence_words
from quran_forced_align.srt import build_rich_records

LOG_FILE = "/content/pipeline_perf_full.log"

def log_msg(msg, f_out=None):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    if f_out:
        f_out.write(formatted + "\n")
        f_out.flush()

def run_profiler():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    f_log = open(LOG_FILE, "w", encoding="utf-8")
    
    log_msg("=" * 90, f_log)
    log_msg("QURAN FORCED ALIGNMENT - COMPREHENSIVE END-TO-END PERFORMANCE PROFILER", f_log)
    log_msg("=" * 90, f_log)
    
    # 0. Environment Diagnostics
    env = verify_colab_environment()
    log_msg(f"Environment: Colab={env['is_colab']}, GPU={env['gpu_name']} ({env['cuda_version']}), ORT_GPU={env['ort_gpu_ok']}", f_log)
    
    # 1. Download / Load Audio
    test_url = "https://media.assabile.com/assabile/recitations_7892537823/mp3/abdel-mohsen-al-obeikan/abdalmhsn-al-bykan-002-al-baqara-234-2232.mp3"
    audio_path = "/content/test_surah2.mp3"
    
    if not os.path.exists(audio_path):
        log_msg(f"[Stage 0/6] Downloading raw recitation audio from {test_url} ...", f_log)
        t_dl0 = time.perf_counter()
        req = urllib.request.Request(test_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(audio_path, "wb") as f:
            f.write(resp.read())
        t_dl = time.perf_counter() - t_dl0
        fsize_mb = os.path.getsize(audio_path) / (1024 * 1024)
        log_msg(f"  Downloaded {fsize_mb:.1f} MB in {t_dl:.3f}s ({fsize_mb/max(0.001, t_dl):.2f} MB/s)", f_log)
    else:
        log_msg(f"[Stage 0/6] Audio already cached at {audio_path} ({os.path.getsize(audio_path)/(1024*1024):.1f} MB)", f_log)
        
    # Model and tokens resolution
    model_path = resolve_model(device="cuda", prefer_fp16=True)
    tokens_path = resolve_tokens()
    log_msg(f"Model Path:  {model_path}", f_log)
    log_msg(f"Tokens Path: {tokens_path}", f_log)
    
    t_pipeline_start = time.perf_counter()
    
    # Stage 1: Reference Building
    t0 = time.perf_counter()
    log_msg("[Stage 1/6] Building Quran phoneme & word-slot reference...", f_log)
    tok2id, id2tok, blank_id, max_token_len = load_tokens(tokens_path)
    combined_token_ids, word_slots = build_combined_reference(2, tok2id, max_token_len, include_istiaatha=False)
    t_stage1 = time.perf_counter() - t0
    log_msg(f"  Stage 1 complete: {len(word_slots)} words, {len(combined_token_ids)} tokens in {t_stage1:.4f}s", f_log)
    
    # Stage 2: Audio Loading + Fbank Feature Extraction
    t0 = time.perf_counter()
    log_msg("[Stage 2/6] Decoding audio to 16kHz float32 & computing 80-dim Fbank features...", f_log)
    t_dec0 = time.perf_counter()
    samples = load_audio_as_wav16k(audio_path)
    t_dec = time.perf_counter() - t_dec0
    audio_sec = len(samples) / 16000.0
    
    t_fb0 = time.perf_counter()
    feats = compute_fbank_features(samples)
    t_fb = time.perf_counter() - t_fb0
    
    t_sil0 = time.perf_counter()
    sil_samples = find_silence_midpoints(samples, 16000)
    sil_frames = [s // 160 for s in sil_samples]
    t_sil = time.perf_counter() - t_sil0
    
    t_stage2 = time.perf_counter() - t0
    log_msg(f"  Stage 2 complete: {audio_sec:.1f}s audio decoded in {t_dec:.3f}s ({audio_sec/max(0.001, t_dec):.0f}x realtime)", f_log)
    log_msg(f"  Fbank frames: {feats.shape[0]} computed in {t_fb:.3f}s ({audio_sec/max(0.001, t_fb):.0f}x realtime)", f_log)
    log_msg(f"  Silence midpoints: {len(sil_frames)} detected in {t_sil:.3f}s", f_log)
    log_msg(f"  Stage 2 Total: {t_stage2:.3f}s", f_log)
    
    # Stage 3: Streaming Zipformer2-CTC GPU Inference
    t0 = time.perf_counter()
    log_msg("[Stage 3/6] Running Batched Streaming Zipformer2-CTC on GPU (intra-surah splitting)...", f_log)
    engine = get_engine("cuda")(model_path)
    log_probs, seconds_per_frame = engine.run_inference_intra_surah_split(feats, sil_frames)
    torch.cuda.synchronize()
    t_stage3 = time.perf_counter() - t0
    log_probs_shape = tuple(log_probs.shape)
    log_msg(f"  Stage 3 complete: log_probs {log_probs_shape} in {t_stage3:.3f}s ({audio_sec/max(0.001, t_stage3):.1f}x realtime!)", f_log)
    
    # Stage 4: GPU Segmented Forced Alignment
    t0 = time.perf_counter()
    log_msg("[Stage 4/6] Running Segmented CTC Forced Alignment on GPU...", f_log)
    ext, path, margins = engine.forced_align_segmented(log_probs, combined_token_ids, blank_id, sil_frames, word_slots)
    torch.cuda.synchronize()
    first_seen, last_seen = frame_spans_from_path(path, len(ext))
    cues = extract_word_frame_spans(word_slots, first_seen, last_seen)
    t_stage4 = time.perf_counter() - t0
    log_msg(f"  Stage 4 complete: {len(cues)} main-pass word cues aligned in {t_stage4:.3f}s ({audio_sec/max(0.001, t_stage4):.1f}x realtime!)", f_log)
    
    # Stage 5: Repeat Detection & Splicing
    t0 = time.perf_counter()
    log_msg("[Stage 5/6] Detecting and locally re-aligning repeated phrases...", f_log)
    min_word_dur_frames = int(0.15 / seconds_per_frame)
    cues = detect_and_fix_repeats(
        engine, cues, log_probs, combined_token_ids, blank_id, ext, path,
        0.15, 2.2, min_word_dur_frames, ayah_final_high_ratio_mult=1.3, confidence_margin=0.6
    )
    t_stage5 = time.perf_counter() - t0
    n_repeats = len([c for c in cues if c.get("is_repeat")])
    log_msg(f"  Stage 5 complete: {len(cues)} total cues ({n_repeats} repeated words spliced) in {t_stage5:.3f}s", f_log)
    
    # Stage 6: Vectorized Confidence Scoring & Output Records
    t0 = time.perf_counter()
    log_msg("[Stage 6/6] Computing per-word acoustic confidence & building rich records...", f_log)
    cues = flag_low_confidence_words(cues, log_probs, ext, path, margins)
    records = build_rich_records(cues, seconds_per_frame, combined_token_ids, id2tok, strip_istiaatha=False)
    t_stage6 = time.perf_counter() - t0
    log_msg(f"  Stage 6 complete: {len(records)} output records in {t_stage6:.4f}s", f_log)
    
    t_pipeline_total = time.perf_counter() - t_pipeline_start
    
    log_msg("=" * 90, f_log)
    log_msg("FINAL PERFORMANCE PROFILE BREAKDOWN", f_log)
    log_msg("=" * 90, f_log)
    log_msg(f"  Audio Recitation Span: {audio_sec:.2f}s ({audio_sec/60:.1f} minutes)", f_log)
    log_msg(f"  Stage 1 (Phonemes & Slots):     {t_stage1:.4f}s  ({100*t_stage1/t_pipeline_total:5.1f}%)", f_log)
    log_msg(f"  Stage 2 (Audio Load & Fbank):   {t_stage2:.4f}s  ({100*t_stage2/t_pipeline_total:5.1f}%)", f_log)
    log_msg(f"  Stage 3 (GPU Zipformer CTC):    {t_stage3:.4f}s  ({100*t_stage3/t_pipeline_total:5.1f}%) -> {audio_sec/max(0.001, t_stage3):.1f}x Realtime", f_log)
    log_msg(f"  Stage 4 (GPU Segmented Align):  {t_stage4:.4f}s  ({100*t_stage4/t_pipeline_total:5.1f}%) -> {audio_sec/max(0.001, t_stage4):.1f}x Realtime", f_log)
    log_msg(f"  Stage 5 (Repeat Detection):     {t_stage5:.4f}s  ({100*t_stage5/t_pipeline_total:5.1f}%)", f_log)
    log_msg(f"  Stage 6 (Confidence Scoring):   {t_stage6:.4f}s  ({100*t_stage6/t_pipeline_total:5.1f}%)", f_log)
    log_msg("-" * 90, f_log)
    log_msg(f"  TOTAL END-TO-END PIPELINE:      {t_pipeline_total:.4f}s", f_log)
    log_msg(f"  REALTIME THROUGHPUT MULTIPLIER: {audio_sec / max(0.001, t_pipeline_total):.1f}x REALTIME", f_log)
    log_msg("=" * 90, f_log)
    
    f_log.close()
    print(f"\nFull performance logs saved to {LOG_FILE}")

if __name__ == "__main__":
    run_profiler()
