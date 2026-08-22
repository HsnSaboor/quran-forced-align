"""Batch 30-second snippet alignment processor for Quran reciters.

Streams the first 30 seconds of recitation audio per surah per reciter,
extracts acoustic features, runs Zipformer CTC forced-alignment,
and generates word-level JSON with timing and repeat flags.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from quran_forced_align.audio import SAMPLE_RATE
from quran_forced_align.features import compute_fbank_features
from quran_forced_align.model_manager import resolve_model, resolve_tokens, resolve_device
from quran_forced_align.engines import get_engine
from quran_forced_align.reference.combined import build_combined_reference
from quran_forced_align.tokenizer import load_tokens
from quran_forced_align.pipeline import _align_from_log_probs
from quran_forced_align.constants import (
    DEFAULT_ANOMALY_LOW_RATIO,
    DEFAULT_ANOMALY_HIGH_RATIO,
    DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
    DEFAULT_REPEAT_CONFIDENCE_MARGIN,
    DEFAULT_MAX_REPEAT_WINDOW_WORDS,
    DEFAULT_TAIL_SILENCE_SEC,
)


def stream_audio_snippet(url_or_path: str, duration_sec: float = 30.0) -> np.ndarray | None:
    """Stream duration_sec seconds of audio via ffmpeg to 16kHz mono float32 PCM."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", "0", "-t", str(duration_sec),
        "-i", url_or_path,
        "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "pipe:1"
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=20, check=True)
        samples = np.frombuffer(proc.stdout, dtype=np.float32)
        if len(samples) < SAMPLE_RATE:
            return None
        return samples
    except Exception:
        return None


def process_snippet(
    surah: int,
    url_or_path: str,
    engine,
    tokens_info: tuple,
    duration_sec: float = 30.0,
    include_istiaatha: bool = False,
    device: str = "cpu",
) -> list[dict] | None:
    """Align the first duration_sec seconds of a surah recitation."""
    samples = stream_audio_snippet(url_or_path, duration_sec=duration_sec)
    if samples is None:
        return None

    tok2id, id2tok, blank_id, max_token_len = tokens_info
    combined_token_ids, word_slots = build_combined_reference(
        surah, tok2id, max_token_len, include_istiaatha=include_istiaatha
    )

    n_frames = int(len(samples) / (SAMPLE_RATE * 0.01))
    max_tokens = min(len(combined_token_ids), max(10, n_frames // 6))
    
    sub_combined_ids = combined_token_ids[:max_tokens]
    sub_word_slots = [w for w in word_slots if w["token_positions"] and w["token_positions"][-1] < max_tokens]

    if not sub_combined_ids or not sub_word_slots:
        return None

    feats = compute_fbank_features(samples, tail_silence_sec=0.5)

    log_probs, seconds_per_frame = engine.run_inference(feats)
    
    def dummy_log(msg): pass
    records = _align_from_log_probs(
        engine,
        log_probs,
        seconds_per_frame,
        sub_combined_ids,
        blank_id,
        sub_word_slots,
        id2tok,
        anomaly_low_ratio=DEFAULT_ANOMALY_LOW_RATIO,
        anomaly_high_ratio=DEFAULT_ANOMALY_HIGH_RATIO,
        ayah_final_high_ratio_mult=DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
        repeat_confidence_margin=DEFAULT_REPEAT_CONFIDENCE_MARGIN,
        max_repeat_window_words=DEFAULT_MAX_REPEAT_WINDOW_WORDS,
        log=dummy_log,
    )
    return records


def main():
    ap = argparse.ArgumentParser(description="Process 30-second openings for Quran reciters")
    ap.add_argument("--json", default="verified_curl_cffi.json", help="Path to verified_curl_cffi.json")
    ap.add_argument("--out-dir", default="snippets_output", help="Output directory for JSON results")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max-reciters", type=int, default=5, help="Number of reciters to process")
    ap.add_argument("--surahs", default="1,2,9,114", help="Comma-separated surahs to process")
    ap.add_argument("--duration", type=float, default=30.0, help="Snippet duration in seconds")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model_path = resolve_model(None, device=device)
    tokens_path = resolve_tokens(None)
    tokens_info = load_tokens(tokens_path)
    engine = get_engine(device)(model_path)

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    verified = data.get("verified", [])[:args.max_reciters]
    surah_list = [int(s.strip()) for s in args.surahs.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"🚀 Processing {len(verified)} reciters across {len(surah_list)} surahs (30s snippets)")
    print(f"   Device: {device.upper()} | Model: {model_path}")

    total_processed = 0
    t0 = time.time()

    for r in verified:
        slug = r.get("slug", "unknown")
        reciter_dir = os.path.join(args.out_dir, slug)
        os.makedirs(reciter_dir, exist_ok=True)

        for surah in surah_list:
            s_str = str(surah)
            if s_str not in r.get("surahs", {}):
                continue
            url = r["surahs"][s_str].get("mp3_url")
            if not url:
                continue

            records = process_snippet(
                surah=surah,
                url_or_path=url,
                engine=engine,
                tokens_info=tokens_info,
                duration_sec=args.duration,
                include_istiaatha=False,
                device=device,
            )

            if records:
                out_file = os.path.join(reciter_dir, f"{surah:03d}_30s.json")
                with open(out_file, "w", encoding="utf-8") as out_f:
                    json.dump(records, out_f, ensure_ascii=False, indent=2)
                print(f"   ✅ [{slug}] Surah {surah:3d} -> {len(records)} words aligned -> {out_file}")
                total_processed += 1
            else:
                print(f"   ⚠️ [{slug}] Surah {surah:3d} -> Failed to decode snippet")

    elapsed = time.time() - t0
    print(f"\n✨ Done! Processed {total_processed} surah snippets in {elapsed:.2f}s (avg: {elapsed/max(1, total_processed):.2f}s per snippet)")


if __name__ == "__main__":
    main()
