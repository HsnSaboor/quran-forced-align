#!/usr/bin/env python3
"""Run Quran Forced Alignment for Mansour Al Salmi (Murattal) - Surahs 1 & 2.

Handles opening preamble (Isti'adha / Bismillah) accurately:
- Surah 1: Aya 1 is Bismillah. Timestamps for Bismillah are aligned to 1:1.
- Surah 2+: Preamble (Bismillah) is bound to Aya 0 for CTC forced alignment,
  then omitted from the output JSON & PB so that 2:1 ("الٓمٓ") starts at its
  true spoken onset without introductory Bismillah audio.

Properly separates:
- `v.words`: EXACTLY 1 entry per canonical word (length = canonical word count, 1-to-1 index).
- `v.segments`: All spoken word instances in chronological order (including repeats with conf=0.9).
- `segments`: [global_word_idx, start_ms, end_ms] mapping cleanly to canonical text without index drift.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort

# Add repo src to sys.path
_repo_src = str(Path(__file__).resolve().parent.parent / "src")
if _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

import quran_transcript as qt
from quran_forced_align.audio import load_audio_as_wav16k
from quran_forced_align.constants import SAMPLE_RATE
from quran_forced_align.features import compute_fbank_features
from quran_forced_align.tokenizer import load_tokens, tokenize_with_char_starts
from quran_forced_align.reference.text import build_text_reference
from quran_forced_align.reference.surah import build_ayah_reference
from quran_forced_align.reference.boundary import _boundary_bridge_rules
from quran_forced_align.onnx_model import run_streaming_log_probs
from quran_forced_align.viterbi import ctc_forced_align, frame_spans_from_path
from quran_forced_align.repeats import detect_and_fix_repeats, extract_word_frame_spans
from quran_forced_align.confidence import flag_low_confidence_words
from quran_forced_align.srt import build_rich_records
from quran_forced_align.corrections import align_spoken_to_canonical, default_registry


# --- Protobuf Serializer ---
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
    payload += write_varint_field(1, surah_id)
    payload += write_varint_field(2, total_duration_ms)
    for v in verses:
        # In Protobuf, segments or words can be serialized; using segments preserves repeat audio spans
        segs = v.get("segments", v.get("words", []))
        ayah_payload = serialize_ayah(v["key"], segs)
        payload += write_length_delimited_field(3, ayah_payload)
    return payload


# --- Acoustic Preamble Probe ---
def probe_preamble_acoustic(log_probs, id2tok, blank_id=0, max_frames=300):
    """Probe the first ~12 seconds of acoustic log_probs for Isti'adha & Bismillah."""
    T = min(len(log_probs), max_frames)
    argmax_ids = np.argmax(log_probs[:T], axis=-1)
    tokens = []
    prev = None
    for tid in argmax_ids:
        tid = int(tid)
        if tid != blank_id and tid != prev:
            tok = id2tok.get(tid, "")
            if tok and tok != "<blank>":
                tokens.append(tok)
        prev = tid
    decoded = "".join(tokens)

    istiaatha_sigs = ("ءَعُ", "عُۥۥذُ", "عُذُ", "ششَي", "طَاانِ", "طَانِ", "جِۦۦم", "جِيم")
    has_istiaatha = any(sig in decoded for sig in istiaatha_sigs)

    bismillah_sigs = ("بِسمِ", "بِس", "للَااهِ", "ررَحمَاانِ", "ررَحِۦۦم", "ررَحِم")
    has_bismillah = any(sig in decoded for sig in bismillah_sigs)

    return has_istiaatha, has_bismillah, decoded


# --- Build Reference with Preamble ---
def build_custom_surah_reference(surah_id: int, has_istiaatha: bool, has_bismillah: bool):
    """Builds per-ayah reference list including exact preamble for forced alignment."""
    refs = []

    # Construct preamble text (Aya 0)
    preamble_parts = []
    if has_istiaatha:
        preamble_parts.append("أَعُوذُ بِٱللَّهِ مِنَ ٱلشَّيْطَانِ ٱلرَّجِيمِ")
    if has_bismillah and surah_id > 1: # For Surah 1, Bismillah is Aya 1:1
        preamble_parts.append("بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ")

    if preamble_parts:
        preamble_text = " ".join(preamble_parts)
        preamble_words = preamble_text.split()
        preamble_ref = build_text_reference(preamble_text, preamble_words, surah_id, 0)
        refs.append(preamble_ref)

    # Number of ayahs in this surah
    n_ayat = qt.Aya(surah_id, 1).get().num_ayat_in_sura
    for aya_idx in range(1, n_ayat + 1):
        refs.append(build_ayah_reference(surah_id, aya_idx))

    return refs


def compile_combined_reference(refs, tok2id, max_token_len, include_boundary_tajweed=True):
    """Compiles list of ayah references into combined token IDs and word slots."""
    combined_token_ids = []
    word_slots = []

    for ref in refs:
        ids, char_starts = tokenize_with_char_starts(ref["phonemes"], tok2id, max_token_len)
        word_slot_base = len(word_slots)
        n_words = len(ref["words"])
        for wi, (ws, we) in enumerate(ref["word_spans"]):
            word_slots.append({
                "word": ref["words"][wi],
                "sura": ref["sura_idx"],
                "aya": ref["aya_idx"],
                "is_ayah_final": wi == n_words - 1,
                "token_positions": [],
                "token_char_idx": [],
                "letters": copy.deepcopy(ref["char_info"][ws:we]),
            })
        for local_tok_idx, cs in enumerate(char_starts):
            wi = ref["phoneme_to_word"][cs] if cs < len(ref["phoneme_to_word"]) else None
            char_idx = ref["phoneme_to_char"][cs] if cs < len(ref["phoneme_to_char"]) else None
            global_pos = len(combined_token_ids) + local_tok_idx
            if wi is not None:
                slot = word_slots[word_slot_base + wi]
                slot["token_positions"].append(global_pos)
                word_start = ref["word_spans"][wi][0]
                slot["token_char_idx"].append(char_idx - word_start)
        combined_token_ids.extend(ids)

    if include_boundary_tajweed:
        for i in range(len(word_slots) - 1):
            if not word_slots[i]["is_ayah_final"]:
                continue
            prev_extra, next_extra = _boundary_bridge_rules(word_slots[i]["word"], word_slots[i + 1]["word"])
            for char_idx, rules in prev_extra:
                word_slots[i]["letters"][char_idx]["boundary_tajweed_rules"] = rules
            for char_idx, rules in next_extra:
                word_slots[i + 1]["letters"][char_idx]["boundary_tajweed_rules"] = rules

    return combined_token_ids, word_slots


class FastCPUEngine:
    def __init__(self, model_path: str, num_threads: int = 4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])

    def run_inference(self, feats):
        return run_streaming_log_probs(self._session, feats)

    def forced_align(self, log_probs, ref_ids, blank_id):
        return ctc_forced_align(log_probs, ref_ids, blank_id)


def process_surah(surah_id: int, base_dir: str, model_path: str, tokens_path: str, num_threads: int = 4):
    t_start = time.time()
    s_str = f"{surah_id:03d}"
    surah_dir = Path(base_dir) / s_str
    audio_path = surah_dir / f"{s_str}.opus"
    if not audio_path.exists():
        audio_path = surah_dir / f"{s_str}.mp3"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file for surah {s_str} not found in {surah_dir}")

    print(f"\n{'='*70}\n[Surah {surah_id:03d}] Processing {audio_path.name} ({audio_path})\n{'='*70}", flush=True)

    # 1. Load tokens & Model
    tok2id, id2tok, blank_id, max_token_len = load_tokens(tokens_path)
    engine = FastCPUEngine(model_path, num_threads=num_threads)

    # 2. Load audio & extract features
    t0 = time.time()
    samples = load_audio_as_wav16k(str(audio_path))
    duration_ms = int(round(len(samples) / SAMPLE_RATE * 1000.0))
    feats = compute_fbank_features(samples)
    print(f"[1/5] Audio loaded: {len(samples)/SAMPLE_RATE:.1f}s ({duration_ms}ms), {feats.shape[0]} fbank frames in {time.time()-t0:.2f}s", flush=True)

    # 3. Acoustic Inference
    t0 = time.time()
    log_probs, seconds_per_frame = engine.run_inference(feats)
    t_inf = time.time() - t0
    print(f"[2/5] Acoustic inference: {log_probs.shape} in {t_inf:.2f}s ({(len(samples)/SAMPLE_RATE)/max(0.001, t_inf):.1f}x realtime)", flush=True)

    # 4. Probe opening preamble
    has_isti, has_bis, intro_text = probe_preamble_acoustic(log_probs, id2tok, blank_id=blank_id)
    print(f"[3/5] Acoustic Preamble Probe: Isti'adha={'DETECTED' if has_isti else 'NO'}, Bismillah={'DETECTED' if has_bis else 'NO'}", flush=True)
    print(f"      Decoded intro: {intro_text[:60]}...", flush=True)

    # 5. Build reference & slots
    t0 = time.time()
    refs = build_custom_surah_reference(surah_id, has_isti, has_bis)
    combined_token_ids, word_slots = compile_combined_reference(refs, tok2id, max_token_len)
    print(f"[4/5] Reference built: {len(refs)} ayahs (incl preamble), {len(word_slots)} words, {len(combined_token_ids)} tokens in {time.time()-t0:.2f}s", flush=True)

    # 6. CTC Trellis Forced Alignment
    t0 = time.time()
    ext, path, margins = engine.forced_align(log_probs, combined_token_ids, blank_id)
    first_seen, last_seen = frame_spans_from_path(path, len(ext))
    cues = extract_word_frame_spans(word_slots, first_seen, last_seen)
    t_viterbi = time.time() - t0
    print(f"[5/5] Viterbi alignment: {len(cues)}/{len(word_slots)} words aligned in {t_viterbi:.2f}s", flush=True)

    # 7. Repeat detection & confidence scoring
    min_word_dur_frames = int(0.05 / seconds_per_frame)
    cues = detect_and_fix_repeats(
        engine, cues, log_probs, combined_token_ids, blank_id, ext, path,
        anomaly_low_ratio=0.15, anomaly_high_ratio=4.0,
        min_word_dur_frames=min_word_dur_frames,
    )
    cues = flag_low_confidence_words(cues, log_probs, ext, path, margins)
    records = build_rich_records(cues, seconds_per_frame, combined_token_ids, id2tok, strip_istiaatha=False)

    # 8. Group records by Ayah (omitting Aya 0 for dataset JSON/PB)
    ayah_records = {}
    repeat_count = 0
    for r in records:
        aya = r.get("aya", 1)
        if aya == 0:
            continue # Omit preamble (Isti'adha / introductory Bismillah) from output verses
        ayah_records.setdefault(aya, []).append(r)

    # 9. Format Canonical Verses & Global Segments
    canon_refs = [r for r in refs if r["aya_idx"] > 0]
    verses = []
    global_word_idx = 1
    flat_segments = []

    for aya_idx, ref_ayah in enumerate(canon_refs, 1):
        canon_words = ref_ayah["words"]
        v_recs = ayah_records.get(aya_idx, [])

        # 1. Build map from local word index (1..N) to global word index in surah
        verse_wi_to_global = {}
        for wi in range(1, len(canon_words) + 1):
            verse_wi_to_global[wi] = global_word_idx
            global_word_idx += 1

        # 2. Map spoken records to canonical words using DP sequence alignment
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

            g_idx = verse_wi_to_global.get(wi, wi)
            flat_segments.append([g_idx, st_ms, et_ms])

        # 3. Canonical words array: exactly 1 entry per canonical word of the ayah
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

        verse_canonical_words.sort(key=lambda w: w["wordIndex"])
        spoken_segments.sort(key=lambda s: s["startTime"])

        v_obj = {
            "verseId": aya_idx,
            "key": f"{surah_id}:{aya_idx}",
            "verseConfidence": 1.0,
            "words": verse_canonical_words,
            "segments": spoken_segments,
        }
        verses.append(v_obj)

    # 10. Metadata & Output Structure
    metadata = {
        "pipeline": "asr_v15.0",
        "surah_id": surah_id,
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

    # 11. Write JSON & Protobuf
    json_path = surah_dir / f"{s_str}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=2)

    pb_bytes = serialize_surah_pb(surah_id, duration_ms, verses)
    pb_path = surah_dir / f"{s_str}.pb"
    with open(pb_path, "wb") as f:
        f.write(pb_bytes)

    t_total = time.time() - t_start
    print(f"\n[DONE] Surah {surah_id:03d} completed in {t_total:.2f}s ({(duration_ms/1000)/t_total:.1f}x realtime)")
    print(f"  -> JSON: {json_path} ({json_path.stat().st_size:,} bytes)")
    print(f"  -> PB:   {pb_path} ({pb_path.stat().st_size:,} bytes)")
    print(f"  -> Verses: {len(verses)}, Canonical Words: {sum(len(v['words']) for v in verses)}, Spoken Segments: {len(flat_segments)}, Repeats: {repeat_count}")
    
    # Print sample verification
    for vi in [0, 1]:
        if vi < len(verses):
            v = verses[vi]
            print(f"  Sample {v['key']}: {v['words'][0]['startTime']}ms - {v['words'][-1]['endTime']}ms ({len(v['words'])} canonical words, {len(v['segments'])} segments)")
            for w in v['words'][:3]:
                print(f"    {w['wordIndex']} {w['text']}: {w['startTime']}ms - {w['endTime']}ms")

    return {
        "surah": surah_id,
        "verses": len(verses),
        "canonical_words": sum(len(v["words"]) for v in verses),
        "segments": len(flat_segments),
        "repeats": repeat_count,
        "duration_sec": duration_ms / 1000.0,
        "elapsed_sec": t_total,
    }


def main():
    base_dir = "/Backup/Quranic-Recitation-Data/Mansour Al Salmi (Murattal)"
    model_path = "/home/zaibi/.cache/quran-forced-align/quran-stt-int8.onnx"
    tokens_path = str(Path(__file__).resolve().parent.parent / "model" / "tokens.txt")

    print(f"====================================================================")
    print(f"  RUNNING FORCED ALIGNMENT FOR MANSOUR AL SALMI — SURAHS 1 & 2")
    print(f"====================================================================")
    print(f"• Base Directory : {base_dir}")
    print(f"• Model Path     : {model_path}")
    print(f"• Tokens Path    : {tokens_path}")

    # Process Surah 1
    res1 = process_surah(1, base_dir, model_path, tokens_path, num_threads=4)

    # Process Surah 2
    res2 = process_surah(2, base_dir, model_path, tokens_path, num_threads=4)

    print(f"\n====================================================================")
    print(f"  ALL COMPLETED SUCCESSFULLY!")
    print(f"====================================================================")
    print(f"Surah 1: {res1['verses']} verses, {res1['canonical_words']} canonical words in {res1['elapsed_sec']:.1f}s")
    print(f"Surah 2: {res2['verses']} verses, {res2['canonical_words']} canonical words, {res2['segments']} segments in {res2['elapsed_sec']:.1f}s ({res2['elapsed_sec']/60:.2f} min)")


if __name__ == "__main__":
    main()
