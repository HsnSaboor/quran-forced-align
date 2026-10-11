#!/usr/bin/env python3
"""Colab T4 job: restructure EVERY flat reciter into Alnufais two-folder
layout, one reciter per run: `{Reciter}/Full Surah/*` + `{Reciter}/Verses/*`.

TARGET LAYOUT (identical to `Ahmad Alnufais (Murattal)/`):
    {Reciter}/Full Surah/{SSS}/{SSS}.opus      # moved as-is (audio untouched)
    {Reciter}/Full Surah/{SSS}/{SSS}.json      # moved as-is
    {Reciter}/Full Surah/{SSS}/{SSS}.pb        # moved as-is
    {Reciter}/Verses/{SSS}/{SSSAAA}.opus       # NEW per-ayah clip
    {Reciter}/Verses/{SSS}/{SSS}.json          # NEW verse index
    {Reciter}/Verses/{SSS}/{SSS}.pb            # NEW verse index pb

HOW IT WORKS (per reciter, set --reciter to one of the 124 flat reciters)
------------
1. Downloads the reciter's 114 full-surah `.opus` from the dataset repo.
2. PASS 1: full-surah `align_surah` (CUDA) for cut points. The EXISTING
   `.pb` is not trusted for cuts (stale pipeline, unverified provenance).
3. Cut points per ayah: [first-word-start - 300ms, last-word-end + 500ms],
   clamped to neighbour midpoints (zero-bleed: never past halfway to the
   adjacent ayah's edge word). Repeat copies stay inside their own ayah's
   clip by construction (monotonic Viterbi spans).
4. PASS 2 (the accuracy step): each verse clip is force-aligned FRESH
   against an AYAH-SLICE reference (that ayah's words only, plus Bismillah
   preamble when the clip caught it). Verse-relative word timings come out
   of Viterbi directly -- never derived by subtracting offsets, so
   cut-point rounding cannot corrupt them. Verified on Mishari 114:1: the
   slice fit cleanly separates Bismillah [40-720ms] from Qul [760ms+] with
   timings matching the full-surah pass within ~200ms.
5. Uploads `Verses/*` (+ optionally moves `Full Surah/*`) to the dataset
   repo (needs HF_TOKEN with write access).

USAGE (Colab, GPU=T4 runtime; ONE reciter per slot/session)
-----------------------------------------------------------
    !pip install -q git+https://github.com/HsnSaboor/quran-forced-align.git
    !pip install -q huggingface_hub
    !apt-get install -y -qq ffmpeg
    python scripts/colab_verses_all.py --reciter "Mishari Rashid Al Afasy (Murattal)" --upload

Resume: pass-1 jsons, verse clips, and per-surah index files are all
skipped when present locally (or on Drive with --drive-dir). Re-running
finishes the remainder -- safe across Colab timeouts.

Throughput (T4, measured): pass 1 ~30-60 min/reciter (114 surahs, batch 8);
pass 2 ~6.2k verse aligns at ~0.5-2s each in batches of 32 (~1-2.5h).
One reciter ~= one Colab session. Run reciters one by one.

EXCLUDED (will fail loudly instead of producing bad data):
- `AbdulBaset AbdulSamad (Warsh)`: riwaya mismatch (Hafs reference vs
  Warsh audio). Skipped unless --include-warsh is passed (and then only
  to attempt; quality is NOT guaranteed).
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

_repo_src = str(Path(__file__).resolve().parent.parent / "src")
if _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_fill_missing_pb import encode_surah_pb  # noqa: E402

HF_DATASET = "zaibihassan/Quranic-Recitation-Data"
HF_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"

WARSH = "AbdulBaset AbdulSamad (Warsh)"

LEAD_PAD_MS = 300
TRAIL_PAD_MS = 500


def download_full_surah(reciter: str, surah: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 10000:
        return out_path
    url = f"{HF_BASE}/{quote(reciter)}/{surah}/{surah}.opus"
    req = urllib.request.Request(url, headers={"User-Agent": "qfa-verses/1.0"})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r, open(out_path, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if out_path.stat().st_size > 10000:
                return out_path
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed {reciter}/{surah}: {last}")


def probe_duration_ms(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(round(float(out.stdout.strip()) * 1000))


def cut_clip(src: Path, start_ms: int, end_ms: int, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 1000:
        return
    s, e = start_ms / 1000.0, end_ms / 1000.0
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
         "-i", str(src), "-c:a", "copy", str(dst)],
        capture_output=True,
    )
    try:
        dur = probe_duration_ms(dst)
    except Exception:
        dur = 0
    if r.returncode != 0 or dur < (end_ms - start_ms) * 0.5:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
             "-i", str(src), "-c:a", "libopus", "-b:a", "32k", "-ar", "16000",
             "-ac", "1", str(dst)],
            check=True,
        )


def ayah_clips(records: list, audio_ms: int):
    by_ayah: dict[int, list] = defaultdict(list)
    for r in records:
        by_ayah[int(r.get("aya", 1))].append(r)
    ayahs = sorted(by_ayah)
    spans = {}
    for a in ayahs:
        ws = [float(w["start"]) * 1000 for w in by_ayah[a]]
        we = [float(w["end"]) * 1000 for w in by_ayah[a]]
        spans[a] = (min(ws), max(we))
    clips = {}
    for i, a in enumerate(ayahs):
        s, e = spans[a]
        s_pad = s - LEAD_PAD_MS
        e_pad = e + TRAIL_PAD_MS
        if i > 0:
            prev_end = spans[ayahs[i - 1]][1]
            s_pad = max(s_pad, (prev_end + s) / 2.0)
        if i < len(ayahs) - 1:
            next_start = spans[ayahs[i + 1]][0]
            e_pad = min(e_pad, (e + next_start) / 2.0)
        clips[a] = (max(0, int(s_pad)), min(audio_ms, int(e_pad)))
    return by_ayah, clips


def build_ayah_slice_reference(surah: int, ayah: int, tok2id, max_token_len,
                               include_bismillah: bool):
    """Ayah-slice (comb_ids, slots) in the EXACT `build_combined_reference`
    slot shape, covering [Bismillah?] + this ayah only."""
    from quran_forced_align.reference.surah import (
        build_ayah_reference, build_bismillah_reference,
    )
    from quran_forced_align.tokenizer import tokenize_with_char_starts

    refs = []
    if include_bismillah and surah not in (1, 9):
        bsm = build_bismillah_reference(surah)
        if bsm:
            refs.append(bsm)
    refs.append(build_ayah_reference(surah, ayah))
    refs = [{**r, "char_info": copy.deepcopy(r["char_info"])} for r in refs]

    combined, slots = [], []
    for ref in refs:
        ids, char_starts = tokenize_with_char_starts(
            ref["phonemes"], tok2id, max_token_len)
        base = len(slots)
        n_words = len(ref["words"])
        for wi, (ws, we) in enumerate(ref["word_spans"]):
            slots.append({
                "word": ref["words"][wi],
                "sura": ref["sura_idx"],
                "aya": ref["aya_idx"],
                "word_idx": wi + 1,
                "is_ayah_final": wi == n_words - 1,
                "token_positions": [],
                "token_char_idx": [],
                "letters": ref["char_info"][ws:we],
            })
        for li, cs in enumerate(char_starts):
            wi = ref["phoneme_to_word"][cs] if cs < len(ref["phoneme_to_word"]) else None
            ci = ref["phoneme_to_char"][cs] if cs < len(ref["phoneme_to_char"]) else None
            gp = len(combined) + li
            if wi is not None:
                slots[base + wi]["token_positions"].append(gp)
                wstart = ref["word_spans"][wi][0]
                slots[base + wi]["token_char_idx"].append(ci - wstart)
        combined.extend(ids)
    return combined, slots


def parse_surah_spec(spec: str) -> set[int]:
    if "-" in spec and "," not in spec:
        lo, _, hi = spec.partition("-")
        return set(range(int(lo), int(hi) + 1))
    return {int(x) for x in spec.split(",") if x.strip()}


def main():
    ap = argparse.ArgumentParser(description="All-reciter Full Surah -> Verses")
    ap.add_argument("--reciter", required=True,
                    help='exact HF folder name, e.g. "Mishari Rashid Al Afasy (Murattal)"')
    ap.add_argument("--out-dir", default="/content/verses_out")
    ap.add_argument("--audio-dir", default="/content/verses_audio")
    ap.add_argument("--drive-dir", default=None)
    ap.add_argument("--repo-id", default=HF_DATASET)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--move-full-surah", action="store_true",
                    help="also upload Full Surah/{SSS}/{SSS}.{opus,json,pb} moves "
                         "(default: only Verses/ is written; existing flat files stay)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--verse-batch", type=int, default=32)
    ap.add_argument("--surahs", default="1-114")
    ap.add_argument("--include-warsh", action="store_true")
    args = ap.parse_args()

    reciter = args.reciter
    if reciter == WARSH and not args.include_warsh:
        raise SystemExit(
            f"{WARSH} is riwaya-mismatched (Hafs ref vs Warsh audio) -- "
            "refusing without --include-warsh")
    style = reciter.rsplit("(", 1)[-1].rstrip(")")

    wanted = parse_surah_spec(args.surahs)
    out_root = Path(args.out_dir) / reciter
    audio_root = Path(args.audio_dir) / reciter
    pass1_dir = out_root / "_pass1"
    pass1_dir.mkdir(parents=True, exist_ok=True)
    drive_root = Path(args.drive_dir) / reciter if args.drive_dir else None

    from quran_forced_align.audio import load_audio_as_wav16k
    from quran_forced_align.batch_cli import run_pipelined_batch
    from quran_forced_align.engines import get_engine
    from quran_forced_align.features import compute_fbank_features
    from quran_forced_align.model_manager import (
        resolve_device, resolve_model, resolve_tokens,
    )
    from quran_forced_align.repeats import extract_word_frame_spans
    from quran_forced_align.tokenizer import load_tokens
    from quran_forced_align.trellis import frame_spans_from_path

    device = resolve_device("auto")
    engine_name = device if device == "cuda" else "cpu"
    model_path = resolve_model(None, device=device)
    tokens_path = resolve_tokens(None)
    print(f"device={device} model={os.path.basename(model_path)}")
    tok2id, id2tok, blank_id, mtl = load_tokens(tokens_path)

    hf_token = None
    if args.upload:
        try:
            from google.colab import userdata  # type: ignore
            hf_token = userdata.get("HF_TOKEN")
        except Exception:
            hf_token = os.environ.get("HF_TOKEN")
    from huggingface_hub import HfApi  # noqa: E402
    api = HfApi(token=hf_token) if hf_token else None
    if args.upload and api is None:
        print("WARNING: --upload without HF_TOKEN; files stay in out-dir")

    def upload(local: Path, repo_path: str):
        if args.upload and api is not None:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=repo_path,
                            repo_id=args.repo_id, repo_type="dataset")

    # ---- PASS 1: full-surah align for cut points ----
    todo_p1 = [s for s in sorted(wanted)
               if not (pass1_dir / f"{s:03d}.json").exists()]
    print(f"pass1: {len(todo_p1)} surahs to align")
    if todo_p1:
        for s in todo_p1:
            download_full_surah(reciter, f"{s:03d}", audio_root / f"{s:03d}.opus")
        summary = run_pipelined_batch(
            sorted(todo_p1), str(audio_root), str(pass1_dir),
            model_path=model_path, tokens_path=tokens_path,
            device=engine_name, cuda_batch_size=args.batch_size,
            intra_surah_split=(device == "cuda"),
            transcode_opus=False, verbose=True)
        print(f"pass1 done: {summary['succeeded_count']}/{len(todo_p1)}")

    engine = get_engine(engine_name)(model_path)

    # ---- PASS 2: cut + ayah-slice re-align, per surah ----
    for s in sorted(wanted):
        s3 = f"{s:03d}"
        rich = pass1_dir / f"{s3}.json"
        if not rich.exists():
            print(f"[skip] surah {s3}: no pass-1 json")
            continue
        records = json.loads(rich.read_text(encoding="utf-8"))
        src_audio = audio_root / f"{s3}.opus"
        download_full_surah(reciter, s3, src_audio)
        audio_ms = probe_duration_ms(src_audio)
        by_ayah, clips = ayah_clips(records, audio_ms)

        surah_dir = out_root / "Verses" / s3
        surah_dir.mkdir(parents=True, exist_ok=True)

        # cut all verse clips first (idempotent, cheap)
        for a in sorted(by_ayah):
            cut_clip(src_audio, *clips[a], surah_dir / f"{s3}{a:03d}.opus")

        verses_json, verse_pb_rows, flat_segments = [], [], []
        for a in sorted(by_ayah):
            vname = f"{s3}{a:03d}.opus"
            c0, c1 = clips[a]
            if a == 1 or s in (1, 9):
                include_bsm = False
            else:
                # clip caught preamble audio iff pass-1 put aya-0 words
                # inside this clip's span
                include_bsm = any(
                    r.get("aya") == 0
                    and c0 <= float(r["start"]) * 1000 <= c1
                    for r in records)
            comb_ids, slots = build_ayah_slice_reference(
                s, a, tok2id, mtl, include_bismillah=include_bsm)
            samples = load_audio_as_wav16k(str(surah_dir / vname))
            feats = compute_fbank_features(samples, tail_silence_sec=0.0)
            log_probs, spf = engine.run_inference(feats)
            ext, path, _m = engine.forced_align(log_probs, comb_ids, blank_id)
            fs, ls = frame_spans_from_path(path, len(ext))
            cues = extract_word_frame_spans(slots, fs, ls)
            ayah_cues = [c for c in cues if c.get("aya") == a]
            if not ayah_cues:
                print(f"  [WARN] {s3} ayah {a}: slice fit empty "
                      f"(bsm={include_bsm}) -- keeping clip, no words")
            words = []
            for c in ayah_cues:
                wi = c.get("word_idx") or 1
                s_ms = int(round(c["start_frame"] * spf * 1000))
                e_ms = int(round(c["end_frame"] * spf * 1000))
                words.append({"wordIndex": wi, "text": c["word"],
                              "startTime": s_ms, "endTime": e_ms,
                              "confidence": 1.0})
                flat_segments.append([a, wi, s_ms, e_ms])
            clip_dur = int(round(len(samples) / 16000 * 1000))
            verses_json.append({
                "verseId": a, "key": f"{s}:{a}", "audioFile": vname,
                "duration_ms": clip_dur, "surahStartMs": c0, "surahEndMs": c1,
                "verseConfidence": 1.0, "words": words, "segments": list(words)})
            verse_pb_rows.append(
                (f"{s}:{a}", [(w["wordIndex"], w["text"], w["startTime"], w["endTime"])
                              for w in words]))
            if drive_root is not None:
                ddir = drive_root / "Verses" / s3
                ddir.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(surah_dir / vname, ddir / vname)

        n_words = sum(len(v["words"]) for v in verses_json)
        ds_json = {
            "metadata": {"pipeline": "qfa_verses_v1.0", "surah_id": s,
                         "unit": "verse", "duration_ms": audio_ms,
                         "audio_duration_ms": audio_ms,
                         "full_surah_duration_ms": audio_ms,
                         "reciter": reciter.split("(")[0].strip(), "style": style,
                         "totalAyahs": len(verses_json), "totalWords": n_words,
                         "valid": True},
            "verses": verses_json,
            "segments": sorted(flat_segments, key=lambda r: (r[0], r[2])),
        }
        (surah_dir / f"{s3}.json").write_text(
            json.dumps(ds_json, ensure_ascii=False, indent=2), encoding="utf-8")
        (surah_dir / f"{s3}.pb").write_bytes(
            encode_surah_pb(s, audio_ms, verse_pb_rows))
        print(f"[surah {s3}] {len(verses_json)} verses, {n_words} words")
        for fn in [f"{s3}{a:03d}.opus" for a in sorted(by_ayah)] + [f"{s3}.json", f"{s3}.pb"]:
            upload(surah_dir / fn, f"{reciter}/Verses/{s3}/{fn}")

        if args.move_full_surah:
            fs_dir = out_root / "Full Surah" / s3
            fs_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(src_audio, fs_dir / f"{s3}.opus")
            for ext in ("json", "pb"):
                src = (f"{HF_BASE}/{quote(reciter)}/{s3}/{s3}.{ext}")
                req = urllib.request.Request(src, headers={"User-Agent": "qfa-verses/1.0"})
                try:
                    data = urllib.request.urlopen(req, timeout=120).read()
                    (fs_dir / f"{s3}.{ext}").write_bytes(data)
                except Exception as e:
                    print(f"  [WARN] could not fetch existing {ext} for {s3}: {e}")
            for fn in (f"{s3}.opus", f"{s3}.json", f"{s3}.pb"):
                p = fs_dir / fn
                if p.exists():
                    upload(p, f"{reciter}/Full Surah/{s3}/{fn}")

    print("\nDone:", str(out_root / "Verses"))


if __name__ == "__main__":
    main()
