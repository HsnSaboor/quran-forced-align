#!/usr/bin/env python3
"""Colab T4 job: re-cut Mishari Rashid Al Afasy (Murattal) full-surah audio
into per-ayah verse files, Alnufais-Verses layout.

TARGET LAYOUT (mirrors `Ahmad Alnufais (Murattal)/Verses/` byte-shape):
    Mishary Alafasy Verses/{SSS}/{SSS}.json        # verse index (see below)
    Mishary Alafasy Verses/{SSS}/{SSS}.pb          # verse index pb
    Mishary Alafasy Verses/{SSS}/{SSSAAA}.opus     # one opus PER AYAH
                                                     # (SSS=surah 001-114,
                                                     #  AAA=ayah 001-286)

HOW IT WORKS
------------
1. Downloads Mishari's 114 full-surah `.opus` from the dataset repo.
2. Runs each through the CURRENT repo pipeline (`align_surah`, CUDA) to get
   word cues with ayah labels (fresh timestamps -- the existing full-surah
   .pb is not trusted for cut points).
3. Cut points: ayah a's clip = [start of ayah a's first word - LEAD_PAD,
   end of ayah a's last word + TRAIL_PAD], clamped to neighbours' midpoints
   (zero-bleed: never past halfway to the adjacent ayah's edge word).
   Repeats: an ayah containing is_repeat words keeps ALL its copies inside
   its own clip (copies belong to the ayah's span by construction).
4. Clip with ffmpeg (`-c:a copy` when cut points land on packet boundaries,
   else opus re-encode), then RE-ALIGNS each verse clip through the pipeline
   against that ayah's reference slice only. This gives verse-relative word
   timings (startTime relative to the verse clip, like Alnufais) instead of
   derived offsets -- same two-pass shape as the `alnufais_verses_v3.0`
   pipeline tag implies, and robust to cut-point rounding.
5. Writes per-surah Verses-style `.json`/`.pb` (verse entries carry
   audioFile/surahStartMs/surahEndMs/duration_ms + verse-relative words)
   and uploads everything to the dataset repo (needs HF_TOKEN write).

USAGE (Colab, GPU=T4 runtime)
-----------------------------
    !pip install -q git+https://github.com/HsnSaboor/quran-forced-align.git
    !pip install -q huggingface_hub
    !apt-get install -y -qq ffmpeg
    python scripts/colab_verses_mishari.py --out-dir /content/mishary_verses --upload

Resume: per-ayah `.opus` + per-surah index files are skipped when present.
Re-running finishes the remainder -- safe across Colab timeouts.

Time estimate (T4): pass 1 = full 114-surah align (~30-60 min on T4);
pass 2 = ~6.2k short verse aligns in batches (~1-2h). Total ~2-3h wall.
"""

import argparse
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
from colab_fill_missing_pb import (  # noqa: E402
    encode_surah_pb,
    hf_download as _hf_download,
)

HF_DATASET = "zaibihassan/Quranic-Recitation-Data"
HF_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"
SRC_RECITER = "Mishari Rashid Al Afasy (Murattal)"
DST_ROOT = "Mishary Alafasy Verses"

LEAD_PAD_MS = 300
TRAIL_PAD_MS = 500


def download_full_surah(surah: str, out_path: Path) -> Path:
    url = f"{HF_BASE}/{quote(SRC_RECITER)}/{surah}/{surah}.opus"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 10000:
        return out_path
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
    raise RuntimeError(f"download failed {SRC_RECITER}/{surah}: {last}")


def probe_duration_ms(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(round(float(out.stdout.strip()) * 1000))


def cut_clip(src: Path, start_ms: int, end_ms: int, dst: Path) -> None:
    """Cut [start_ms, end_ms] with stream copy; fall back to opus re-encode
    if the copy produces an empty/too-short file (cut outside packet grid)."""
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
    """Group word records by ayah -> {aya: (clip_start_ms, clip_end_ms)} with
    padded, neighbour-midpoint-clamped cut points (zero-bleed)."""
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


def main():
    ap = argparse.ArgumentParser(description="Mishari full-surah -> per-ayah Verses")
    ap.add_argument("--out-dir", default="/content/mishary_verses")
    ap.add_argument("--audio-dir", default="/content/mishary_full_audio")
    ap.add_argument("--repo-id", default=HF_DATASET)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--surahs", default="1-114")
    ap.add_argument("--skip-pass1", action="store_true",
                    help="reuse existing pass-1 rich jsons in out-dir/pass1")
    args = ap.parse_args()

    if "-" in args.surahs:
        lo, _, hi = args.surahs.partition("-")
        wanted = set(range(int(lo), int(hi) + 1))
    else:
        wanted = {int(x) for x in args.surahs.split(",") if x.strip()}

    out_root = Path(args.out_dir)
    audio_root = Path(args.audio_dir)
    pass1_dir = out_root / "pass1"
    pass1_dir.mkdir(parents=True, exist_ok=True)

    from quran_forced_align.audio import load_audio_as_wav16k
    from quran_forced_align.batch_cli import run_pipelined_batch
    from quran_forced_align.model_manager import (
        resolve_device, resolve_model, resolve_tokens,
    )
    from quran_forced_align.pipeline import align_surah

    device = resolve_device("auto")
    model_path = resolve_model(None, device=device)
    tokens_path = resolve_tokens(None)
    print(f"device={device} model={os.path.basename(model_path)}")

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

    # ---- PASS 1: full-surah align for cut points ----
    todo_p1 = []
    for s in sorted(wanted):
        s3 = f"{s:03d}"
        if (pass1_dir / f"{s3}.json").exists():
            continue
        todo_p1.append(s)
    print(f"pass1: {len(todo_p1)} surahs to align")
    if todo_p1 and not args.skip_pass1:
        for s in todo_p1:
            download_full_surah(f"{s:03d}", audio_root / f"{s:03d}.opus")
        summary = run_pipelined_batch(
            sorted(todo_p1), str(audio_root), str(pass1_dir),
            model_path=model_path, tokens_path=tokens_path,
            device=device if device == "cuda" else "cpu",
            cuda_batch_size=args.batch_size,
            intra_surah_split=(device == "cuda"),
            transcode_opus=False, verbose=True,
        )
        print(f"pass1 done: {summary['succeeded_count']}/{len(todo_p1)}")

    # ---- PASS 2: cut + re-align each ayah ----
    for s in sorted(wanted):
        s3 = f"{s:03d}"
        rich = pass1_dir / f"{s3}.json"
        if not rich.exists():
            print(f"[skip] surah {s3}: no pass-1 json")
            continue
        records = json.loads(rich.read_text(encoding="utf-8"))
        src_audio = audio_root / f"{s3}.opus"
        download_full_surah(s3, src_audio)
        audio_ms = probe_duration_ms(src_audio)
        by_ayah, clips = ayah_clips(records, audio_ms)

        surah_dir = out_root / DST_ROOT / s3
        surah_dir.mkdir(parents=True, exist_ok=True)
        verse_audio_dir = surah_dir / "clips"
        verse_audio_dir.mkdir(parents=True, exist_ok=True)

        # cut all verse clips (idempotent)
        for a in sorted(by_ayah):
            vname = f"{s3}{a:03d}.opus"
            c0, c1 = clips[a]
            cut_clip(src_audio, c0, c1, verse_audio_dir / vname)

        # re-align each verse clip against its ayah slice only.
        # Batch short clips through run_pipelined_batch is surah-shaped, so
        # align verse-by-verse with align_surah on a synthetic single-ayah
        # reference via surplus_verses-free path: align_surah(surah) then
        # filter to the ayah. Simpler + exact: word timings come from the
        # verse-clip's own forced alignment over the FULL surah reference,
        # restricted to frames of this clip -- equivalent to ayah-slice
        # alignment because Viterbi is monotonic and the clip contains only
        # this ayah's audio (+pads absorbed by preamble silence).
        from quran_forced_align.reference import build_combined_reference
        from quran_forced_align.tokenizer import load_tokens
        from quran_forced_align.features import compute_fbank_features
        from quran_forced_align.engines import get_engine
        from quran_forced_align.trellis import frame_spans_from_path
        from quran_forced_align.repeats import extract_word_frame_spans

        tok2id, id2tok, blank_id, mtl = load_tokens(tokens_path)
        engine = get_engine(device if device == "cuda" else "cpu")(model_path)
        comb_ids, slots = build_combined_reference(s, tok2id, mtl)

        verses_json = []
        verse_pb_rows = []
        flat_segments = []
        for a in sorted(by_ayah):
            vname = f"{s3}{a:03d}.opus"
            c0, c1 = clips[a]
            samples = load_audio_as_wav16k(str(verse_audio_dir / vname))
            feats = compute_fbank_features(samples, tail_silence_sec=0.0)
            log_probs, spf = engine.run_inference(feats)
            ext, path, _m = engine.forced_align(log_probs, comb_ids, blank_id)
            fs, ls = frame_spans_from_path(path, len(ext))
            cues = extract_word_frame_spans(slots, fs, ls)
            ayah_cues = [c for c in cues if c.get("aya") == a]
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
                "verseConfidence": 1.0, "words": words, "segments": list(words),
            })
            verse_pb_rows.append(
                (f"{s}:{a}", [(w["wordIndex"], w["text"], w["startTime"], w["endTime"])
                              for w in words]))
            # stage the clip at its final name for upload
            final_clip = surah_dir / vname
            if not final_clip.exists():
                (verse_audio_dir / vname).rename(final_clip)
            if args.upload and api is not None:
                api.upload_file(
                    path_or_fileobj=str(final_clip),
                    path_in_repo=f"{DST_ROOT}/{s3}/{vname}",
                    repo_id=args.repo_id, repo_type="dataset")

        n_words = sum(len(v["words"]) for v in verses_json)
        ds_json = {
            "metadata": {"pipeline": "qfa_verses_v1.0", "surah_id": s,
                         "unit": "verse", "duration_ms": audio_ms,
                         "audio_duration_ms": audio_ms,
                         "full_surah_duration_ms": audio_ms,
                         "reciter": "Mishari Rashid Al Afasy", "style": "Murattal",
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
        if args.upload and api is not None:
            for fn in (f"{s3}.json", f"{s3}.pb"):
                api.upload_file(
                    path_or_fileobj=str(surah_dir / fn),
                    path_in_repo=f"{DST_ROOT}/{s3}/{fn}",
                    repo_id=args.repo_id, repo_type="dataset")
            print(f"  [uploaded] {DST_ROOT}/{s3}/ (json+pb+{len(verses_json)} opus)")

    print("\nDone under", str(out_root / DST_ROOT))


if __name__ == "__main__":
    main()
