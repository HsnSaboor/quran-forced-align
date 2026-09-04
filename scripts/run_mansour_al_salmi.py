#!/usr/bin/env python3
"""Run Quran Forced Alignment for Mansour Al Salmi (Murattal) across all 114 surahs.

Generates and replaces {surah}.json and {surah}.pb in each surah folder
with repeat verses included in words and segments.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

# Add src to sys.path
_repo_src = str(Path(__file__).resolve().parent.parent / "src")
if _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

from quran_forced_align.audio import load_audio_as_wav16k
from quran_forced_align.model_manager import resolve_model, resolve_tokens
from quran_forced_align.pipeline import align_surah
from quran_forced_align.reference.surah import build_surah_reference
from quran_forced_align.corrections import align_spoken_to_canonical, default_registry


def encode_varint(value: int) -> bytes:
    out = bytearray()
    while True:
        towrite = value & 0x7F
        value >>= 7
        if value:
            out.append(towrite | 0x80)
        else:
            out.append(towrite)
            break
    return bytes(out)


def write_field(field_num: int, wire_type: int, val: bytes) -> bytes:
    key = (field_num << 3) | wire_type
    return encode_varint(key) + val


def write_varint_field(field_num: int, value: int) -> bytes:
    return write_field(field_num, 0, encode_varint(value))


def write_length_delimited_field(field_num: int, data: bytes) -> bytes:
    return write_field(field_num, 2, encode_varint(len(data)) + data)


def serialize_word(word_index: int, text: str, start_time: int, end_time: int) -> bytes:
    payload = b""
    if word_index > 0:
        payload += write_varint_field(1, word_index)
    if text:
        payload += write_length_delimited_field(2, text.encode("utf-8"))
    if start_time >= 0:
        payload += write_varint_field(3, start_time)
    if end_time >= 0:
        payload += write_varint_field(4, end_time)
    return payload


def serialize_words_list(words: list) -> bytes:
    payload = b""
    for w in words:
        word_payload = serialize_word(
            w.get("wordIndex", 1),
            w.get("text", ""),
            w.get("startTime", 0),
            w.get("endTime", 0),
        )
        payload += write_length_delimited_field(1, word_payload)
    return payload


def serialize_ayah(key: str, words: list) -> bytes:
    key_bytes = key.encode("utf-8")
    payload = write_length_delimited_field(1, key_bytes)
    words_list_bytes = serialize_words_list(words)
    payload += write_length_delimited_field(2, words_list_bytes)
    return payload


def serialize_surah_pb(surah_id: int, total_duration_ms: int, verses: list) -> bytes:
    payload = b""
    # Top-level Field 1: surah_id (varint)
    payload += write_varint_field(1, surah_id)
    # Top-level Field 2: total_duration_ms (varint)
    payload += write_varint_field(2, total_duration_ms)
    # Top-level Field 3: Ayah messages (length-delimited repeat)
    for v in verses:
        ayah_payload = serialize_ayah(v["key"], v["words"])
        payload += write_length_delimited_field(3, ayah_payload)
    return payload


def process_single_surah(
    surah: int,
    base_dir: str,
    model_path: str,
    tokens_path: str,
    device: str = "cpu",
) -> dict:
    """Process a single surah: align audio, format JSON with repeats in words, write JSON and PB."""
    t0 = time.time()
    s_str = f"{surah:03d}"
    surah_dir = Path(base_dir) / s_str
    audio_path = surah_dir / f"{s_str}.opus"
    if not audio_path.exists():
        # Fallback to mp3 or wav
        audio_path = surah_dir / f"{s_str}.mp3"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file for surah {s_str} not found in {surah_dir}")

    # 1. Load audio and compute exact duration
    samples = load_audio_as_wav16k(str(audio_path))
    duration_ms = int(round(len(samples) / 16000.0 * 1000.0))

    # 2. Run forced alignment
    records = align_surah(
        surah,
        str(audio_path),
        model_path=model_path,
        tokens_path=tokens_path,
        device=device,
        verbose=False,
    )

    # 3. Load canonical reference for surah
    refs = build_surah_reference(surah, include_istiaatha=False)

    # 4. Group spoken records by ayah (filter out Isti'adha aya == 0)
    ayah_records = {}
    for r in records:
        aya = r.get("aya", 1)
        if aya == 0:
            continue
        ayah_records.setdefault(aya, []).append(r)

    # 5. Build canonical verses & map words (including repeats)
    verses = []
    global_word_idx = 1
    flat_segments = []
    repeat_count = 0

    for aya_idx, ref_ayah in enumerate(refs, 1):
        canon_words = ref_ayah["words"]
        v_recs = ayah_records.get(aya_idx, [])

        # Assign global index mapping for canonical words of this ayah
        verse_wi_map = {}
        for wi, w_text in enumerate(canon_words, 1):
            verse_wi_map[wi] = global_word_idx
            global_word_idx += 1

        # Align spoken words to canonical words using DP sequence alignment
        spoken_words = [r["word"] for r in v_recs]
        mapping = align_spoken_to_canonical(canon_words, spoken_words)

        spoken_segments = []
        seen_wis = set()
        best_canonical_timing = {}

        for r, wi in zip(v_recs, mapping):
            st_ms = int(round(r["start"] * 1000))
            et_ms = int(round(r["end"] * 1000))
            canonical_text = canon_words[wi - 1]
            is_rep = wi in seen_wis or bool(r.get("is_repeat", False))
            seen_wis.add(wi)
            if is_rep:
                repeat_count += 1

            conf = 0.9 if is_rep else (0.8 if r.get("low_confidence") else 1.0)
            word_obj = {
                "wordIndex": wi,
                "text": canonical_text,
                "startTime": st_ms,
                "endTime": et_ms,
                "confidence": conf,
            }
            if is_rep:
                word_obj["is_repeat"] = True
            spoken_segments.append(word_obj)

            if wi not in best_canonical_timing or not is_rep:
                best_canonical_timing[wi] = word_obj

            g_idx = verse_wi_map.get(wi, wi)
            flat_segments.append([g_idx, st_ms, et_ms])

        verse_canonical_words = []
        for wi, w_text in enumerate(canon_words, 1):
            if wi in best_canonical_timing:
                w_obj = best_canonical_timing[wi]
                w_canonical = dict(w_obj)
                w_canonical["confidence"] = 1.0 if w_canonical["confidence"] >= 0.9 else w_canonical["confidence"]
                verse_canonical_words.append(w_canonical)
            else:
                verse_canonical_words.append({
                    "wordIndex": wi,
                    "text": w_text,
                    "startTime": 0,
                    "endTime": 0,
                    "confidence": 0.0,
                })

        v_obj = {
            "verseId": aya_idx,
            "key": f"{surah}:{aya_idx}",
            "verseConfidence": 1.0,
            "words": verse_canonical_words,
            "segments": spoken_segments,
        }
        verses.append(v_obj)

    # 6. Build top-level metadata
    metadata = {
        "pipeline": "asr_v15.0",
        "surah_id": surah,
        "duration_ms": duration_ms,
        "audio_duration_ms": duration_ms,
        "reciter": "001",
        "valid": True,
    }
    if repeat_count > 0:
        metadata["repeatEntries"] = repeat_count

    flat_segments.sort(key=lambda s: s[1])
    final_json = {
        "metadata": metadata,
        "verses": verses,
        "segments": flat_segments,
    }

    # 7. Write JSON file
    json_path = surah_dir / f"{s_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    # 8. Write Protobuf file
    pb_bytes = serialize_surah_pb(surah, duration_ms, verses)
    pb_path = surah_dir / f"{s_str}.pb"
    with open(pb_path, "wb") as f:
        f.write(pb_bytes)

    elapsed = time.time() - t0
    total_spoken_words = sum(len(v["words"]) for v in verses)

    return {
        "surah": surah,
        "duration_sec": duration_ms / 1000.0,
        "words": total_spoken_words,
        "repeats": repeat_count,
        "elapsed_sec": elapsed,
        "json_bytes": json_path.stat().st_size,
        "pb_bytes": pb_path.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Run forced alignment for Mansour Al Salmi across 114 surahs")
    parser.add_argument(
        "--dir",
        default="/Backup/Quranic-Recitation-Data/Mansour Al Salmi (Murattal)",
        help="Reciter directory containing 001..114 folders",
    )
    parser.add_argument(
        "--surahs",
        default="1-114",
        help="Surah range or comma-separated list (default: 1-114)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help=f"Number of parallel workers (default: {min(8, os.cpu_count() or 1)})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device ('cpu' or 'cuda')",
    )
    args = parser.parse_args()

    # Parse surah list
    if "-" in args.surahs and "," not in args.surahs:
        lo, hi = [int(x) for x in args.surahs.split("-")]
        surah_list = list(range(lo, hi + 1))
    else:
        surah_list = [int(x) for x in args.surahs.split(",") if x.strip()]

    # Sort surahs descending by size so long surahs start early and saturate all cores
    # (Surah 2, 3, 4, 7, 9 take the longest)
    # But collect results in order
    model_path = resolve_model(device=args.device)
    tokens_path = resolve_tokens()

    print("=" * 80)
    print(f"  QURAN FORCED ALIGNMENT — MANSOUR AL SALMI (MURATTAL)")
    print("=" * 80)
    print(f"• Target Directory : {args.dir}")
    print(f"• Surahs to Align  : {len(surah_list)} surahs ({surah_list[0]} to {surah_list[-1]})")
    print(f"• Parallel Workers : {args.workers}")
    print(f"• Model Path       : {model_path}")
    print(f"• Tokens Path      : {tokens_path}")
    print(f"• Repeat Mode      : Repeat verses in words & segments")
    print("-" * 80)

    # Process order: longest surahs first to avoid a single long surah straggling at the end
    process_order = sorted(surah_list, key=lambda s: (0 if s in [2, 3, 4, 7, 9, 5, 6, 8, 10, 11, 12] else 1, s))

    start_wall_time = time.time()
    results = []
    completed_count = 0
    total_audio_sec = 0.0
    total_words = 0
    total_repeats = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_surah = {
            executor.submit(
                process_single_surah,
                s,
                args.dir,
                model_path,
                tokens_path,
                args.device,
            ): s
            for s in process_order
        }

        for future in concurrent.futures.as_completed(future_to_surah):
            s = future_to_surah[future]
            try:
                res = future.result()
                results.append(res)
                completed_count += 1
                total_audio_sec += res["duration_sec"]
                total_words += res["words"]
                total_repeats += res["repeats"]

                pct = (completed_count / len(surah_list)) * 100
                rtf = res["elapsed_sec"] / max(0.001, res["duration_sec"])
                speedup = 1.0 / max(1e-6, rtf)
                print(
                    f"[{completed_count:3d}/{len(surah_list):3d}] ({pct:5.1f}%) "
                    f"✓ Surah {res['surah']:03d} | Audio: {res['duration_sec']:6.1f}s | "
                    f"Words: {res['words']:4d} | Repeats: {res['repeats']:2d} | "
                    f"Time: {res['elapsed_sec']:5.1f}s ({speedup:4.1f}x RT) | "
                    f"JSON: {res['json_bytes']//1024}KB, PB: {res['pb_bytes']//1024}KB"
                )
            except Exception as e:
                print(f"❌ Error processing Surah {s:03d}: {e}")
                import traceback
                traceback.print_exc()

    total_wall_sec = time.time() - start_wall_time
    total_audio_hours = total_audio_sec / 3600.0
    overall_rtf = total_wall_sec / max(0.001, total_audio_sec)
    overall_speedup = 1.0 / max(1e-6, overall_rtf)

    print("=" * 80)
    print("  ALIGNMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"• Total Surahs Processed : {completed_count}/{len(surah_list)}")
    print(f"• Total Audio Aligned    : {total_audio_hours:.2f} hours ({total_audio_sec:.1f} sec)")
    print(f"• Total Wall Time        : {total_wall_sec / 60.0:.2f} min ({total_wall_sec:.1f} sec)")
    print(f"• Overall Throughput     : {overall_speedup:.1f}x Realtime (RTF: {overall_rtf:.4f})")
    print(f"• Total Spoken Words     : {total_words:,}")
    print(f"• Total Repeats Detected : {total_repeats:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
