#!/usr/bin/env python3
"""Colab T4 job: align the 123 HF slots missing `.pb` timestamps and emit
dataset-ready `.pb` + `.json` files.

WHAT IT DOES
------------
1. Downloads the 123 missing-slot `.opus` files from
   `zaibihassan/Quranic-Recitation-Data` (114x AbdulBaset AbdulSamad Warsh
   + 6x Ahmed Abdel Razeq Nasr + 2x Youssef Al Aidarous + 1x Mustafa al-Azawi).
2. Runs each through the CURRENT repo pipeline (`align_surah`, CUDA, batch 8,
   intra-surah split) -- identical code path to the committed HEAD.
3. Encodes word cues to the EXACT on-disk `.pb` schema (byte-verified against
   the existing `Mustafa al-Azawi 009.pb`: top-level field1=surah_id varint,
   field2=duration_ms varint, field3=ayah msgs{field1=key "s:a",
   field2=words{field1=word msgs{1=word_idx_1based,2=text,3=start_ms,4=end_ms}}})
   plus the sibling `.json` ({metadata, verses, segments}) the repo already
   writes next to every `.pb`.
4. Uploads `*.pb` (+`*.json`) back to the dataset repo (needs a `HF_TOKEN`
   Colab secret with write access).

USAGE (Colab, GPU=T4 runtime)
-----------------------------
    !pip install -q git+https://github.com/HsnSaboor/quran-forced-align.git
    !pip install -q huggingface_hub
    # optional: mount Drive for a resume cache across sessions
    python scripts/colab_fill_missing_pb.py --out-dir /content/missing_pb
        --upload --repo-id zaibihassan/Quranic-Recitation-Data

Resume: completed surahs are skipped if their `.pb` already exists in
`--out-dir` (or on Drive if `--drive-dir` is given). Re-running the same
command finishes the remainder -- safe across Colab timeouts.

Time estimate (T4, measured pipeline RTF ~0.02-0.05x on long surahs):
~5-6h audio total -> roughly 1.5-3h wall, plus Warsh QA spot-checks.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

_repo_src = str(Path(__file__).resolve().parent.parent / "src")
if _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

HF_DATASET = "zaibihassan/Quranic-Recitation-Data"
HF_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"

# The 123 slots with audio but no .pb (audited 2026-10-11 from HF siblings).
# ON HOLD (do NOT run through this script without a human decision):
# - AbdulBaset AbdulSamad Warsh x114: audio is RIWAYA WARSH (verified: Warsh
#   001 decodes MALIKI YAWM where Hafs reads MAALIKI, and the reciter audibly
#   says Maliki). The pipeline reference text is Hafs everywhere, so
#   forced-aligning Warsh audio against a Hafs reference yields wrong word
#   identities with confident-looking timestamps. Needs a Warsh reference
#   text plus Warsh phonetizer first; then the same pb encoder here works.
# - Mustafa al-Azawi 010: the HF 010 opus (177s) is NOT full Surah Yunus
#   (109 ayahs, 1833 words, needs 14k+ frames; the file yields only 4452).
#   It head-decodes as Yunus ayah 1 but ends after 3 min: a partial or
#   truncated upload. Options: (a) re-source full Surah 10 audio and align,
#   or (b) align the partial file against ayahs 1..N with surplus trimming.
#   Either way NOT a blind align_surah(10).

MISSING = {
    # Durations check out as FULL surahs at a fast recitation rate (Nasr
    # 002 5.4min/286 ayahs, 003 4.2min/200 ayahs, 006 7.3min/165 ayahs,
    # Aidarous 007 14.7min/206 ayahs ~ 4.3s/ayah -- fast murattal but
    # self-consistent). Head-decodes start mid-text because these reciters
    # skip Bismillah / start directly (pipeline auto-detects preamble).
    # Full align_surah per slot; the trellis itself rejects truncation
    # (raises if audio too short) so a partial file can never silently
    # produce a .pb -- failures stay missing, loudly.
    "Ahmed Abdel Razeq Nasr (Murattal)": ["002", "003", "006", "022", "026", "028"],
    "Youssef Al Aidarous (Murattal)": ["006", "007"],
}
assert sum(len(v) for v in MISSING.values()) == 8, "expected 8 runnable slots"


# --------------------------------------------------------------------------
# .pb encoder: byte-compatible with the existing on-disk schema (verified
# against Mustafa al-Azawi 009.pb: field1=surah_id, field2=duration_ms,
# field3+=ayah{1:key "s:a", 2:words{1+=word{1:idx1,2:text,3:from_ms,4:to_ms}}}).
# --------------------------------------------------------------------------

def _varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def _field(num: int, wire: int, payload: bytes) -> bytes:
    return _varint((num << 3) | wire) + payload


def _vi(num: int, v: int) -> bytes:
    return _field(num, 0, _varint(v))


def _ld(num: int, data: bytes) -> bytes:
    return _field(num, 2, _varint(len(data)) + data)


def encode_word(word_idx_1based: int, text: str, start_ms: int, end_ms: int) -> bytes:
    p = b""
    if word_idx_1based > 0:
        p += _vi(1, word_idx_1based)
    if text:
        p += _ld(2, text.encode("utf-8"))
    if start_ms >= 0:
        p += _vi(3, start_ms)
    if end_ms >= 0:
        p += _vi(4, end_ms)
    return p


def encode_surah_pb(surah_id: int, duration_ms: int, verses: list) -> bytes:
    """verses: list of (key, [(word_idx_1based, text, start_ms, end_ms), ...])."""
    p = _vi(1, surah_id) + _vi(2, duration_ms)
    for key, words in verses:
        words_blob = b"".join(_ld(1, encode_word(wi, t, s, e)) for wi, t, s, e in words)
        ayah = _ld(1, key.encode("utf-8")) + _ld(2, words_blob)
        p += _ld(3, ayah)
    return p


# --------------------------------------------------------------------------
# records (.json rich) -> dataset (.pb + .json) conversion
# --------------------------------------------------------------------------

def records_to_dataset(records: list, surah: int, audio_ms: int):
    """Group word records by ayah; repeat copies are EXTRA word entries
    (same word_idx, own span) -- matches existing .pb files which list the
    repeated word twice with distinct timestamps.

    Returns (pb_bytes, json_obj). Times are ms relative to surah audio start.
    """
    verses: dict[str, list] = {}
    order: list[str] = []
    flat_segments: list = []
    max_end = 0
    for r in records:
        aya = r.get("aya", 1)
        key = f"{surah}:{aya}"
        wi = r.get("word_idx") or 1
        s_ms = int(round(float(r["start"]) * 1000))
        e_ms = int(round(float(r["end"]) * 1000))
        max_end = max(max_end, e_ms)
        entry = (wi, r.get("word", ""), s_ms, e_ms)
        if key not in verses:
            verses[key] = []
            order.append(key)
        verses[key].append(entry)
        flat_segments.append([wi, s_ms, e_ms])
    flat_segments.sort(key=lambda s: s[1])
    duration_ms = max(audio_ms, max_end)
    pb = encode_surah_pb(surah, duration_ms, [(k, verses[k]) for k in order])
    n_rep = sum(1 for r in records if r.get("is_repeat"))
    json_obj = {
        "metadata": {
            "pipeline": "quran-forced-align",
            "surah_id": surah,
            "duration_ms": duration_ms,
            "audio_duration_ms": audio_ms,
            "valid": True,
            "repeatEntries": n_rep,
            "words": len(records),
        },
        "verses": [
            {
                "verseId": int(k.split(":")[1]),
                "key": k,
                "words": [
                    {"wordIndex": wi, "text": t, "startTime": s, "endTime": e}
                    for wi, t, s, e in verses[k]
                ],
            }
            for k in order
        ],
        "segments": flat_segments,
    }
    return pb, json_obj


# --------------------------------------------------------------------------
# download / align / upload
# --------------------------------------------------------------------------

def hf_download(reciter: str, surah: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 10000:
        return out_path
    url = f"{HF_BASE}/{quote(reciter)}/{surah}/{surah}.opus"
    req = urllib.request.Request(url, headers={"User-Agent": "qfa-fill-missing/1.0"})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(out_path, "wb") as f:
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


def main():
    ap = argparse.ArgumentParser(description="Align the 123 HF slots missing .pb")
    ap.add_argument("--out-dir", default="/content/missing_pb")
    ap.add_argument("--audio-dir", default="/content/missing_audio")
    ap.add_argument("--drive-dir", default=None,
                    help="optional Drive mirror for resume across sessions")
    ap.add_argument("--repo-id", default=HF_DATASET)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--only", default=None,
                    help="subset: 'Reciter Name:002,003' (repeatable via ';')")
    ap.add_argument("--skip-warsh", action="store_true",
                    help="HOLD the Warsh reciter (riwaya mismatch, see docstring)")
    args = ap.parse_args()

    jobs: list[tuple[str, str]] = []
    if args.only:
        for chunk in args.only.split(";"):
            rec, _, surahs = chunk.partition(":")
            jobs += [(rec.strip(), s.strip()) for s in surahs.split(",") if s.strip()]
    else:
        for rec, surahs in MISSING.items():
            if args.skip_warsh and "Warsh" in rec:
                continue
            jobs += [(rec, s) for s in surahs]

    out_root = Path(args.out_dir)
    audio_root = Path(args.audio_dir)
    drive_root = Path(args.drive_dir) if args.drive_dir else None

    todo = []
    for rec, s in jobs:
        pb_path = out_root / rec / s / f"{s}.pb"
        if pb_path.exists() and pb_path.stat().st_size > 100:
            print(f"[skip] {rec} {s} already done")
            continue
        todo.append((rec, s))
    print(f"{len(todo)}/{len(jobs)} slots remaining")

    from quran_forced_align.audio import load_audio_as_wav16k
    from quran_forced_align.batch_cli import run_pipelined_batch
    from quran_forced_align.model_manager import (
        resolve_device, resolve_model, resolve_tokens,
    )

    device = resolve_device("auto")
    model_path = resolve_model(None, device=device)
    tokens_path = resolve_tokens(None)
    print(f"device={device} model={os.path.basename(model_path)}")

    # group by reciter so each run_pipelined_batch call shares one audio dir
    from collections import defaultdict
    by_rec: dict[str, list[str]] = defaultdict(list)
    for rec, s in todo:
        by_rec[rec].append(s)

    hf_token = None
    if args.upload:
        try:
            from google.colab import userdata  # type: ignore
            hf_token = userdata.get("HF_TOKEN")
        except Exception:
            hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("WARNING: --upload without HF_TOKEN; files will be left in out-dir")

    from huggingface_hub import HfApi  # noqa: E402
    api = HfApi(token=hf_token) if hf_token else None

    for rec, surahs in by_rec.items():
        s_ints = sorted(int(s) for s in surahs)
        rec_audio = audio_root / rec
        rec_audio.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {rec}: downloading {len(s_ints)} surah(s) ===")
        ok_dirs = []
        for s in s_ints:
            s3 = f"{s:03d}"
            try:
                hf_download(rec, s3, rec_audio / f"{s3}.opus")
                ok_dirs.append(s)
            except Exception as e:
                print(f"  [download FAIL] {s3}: {e}")
        if not ok_dirs:
            continue
        print(f"=== aligning {rec} ({len(ok_dirs)} surahs, cuda batch {args.batch_size}) ===")
        rec_out = out_root / rec
        summary = run_pipelined_batch(
            ok_dirs, str(rec_audio), str(rec_out),
            model_path=model_path, tokens_path=tokens_path,
            device=device if device == "cuda" else "cpu",
            cuda_batch_size=args.batch_size,
            intra_surah_split=(device == "cuda"),
            transcode_opus=False,
            verbose=True,
        )
        print(f"aligned {summary['succeeded_count']}/{len(ok_dirs)} "
              f"words={summary['total_words']:,} repeats={summary['total_repeats']:,}")

        # convert each surah's rich .json -> dataset .pb + dataset .json
        for s in ok_dirs:
            s3 = f"{s:03d}"
            rich_json = rec_out / f"{s3}.json"
            if not rich_json.exists():
                print(f"  [convert SKIP] {s3}: no rich json (alignment failed?)")
                continue
            records = json.loads(rich_json.read_text(encoding="utf-8"))
            samples = load_audio_as_wav16k(str(rec_audio / f"{s3}.opus"))
            audio_ms = int(round(len(samples) / 16000 * 1000))
            pb_bytes, ds_json = records_to_dataset(records, s, audio_ms)
            surah_dir = out_root / rec / s3
            surah_dir.mkdir(parents=True, exist_ok=True)
            (surah_dir / f"{s3}.pb").write_bytes(pb_bytes)
            (surah_dir / f"{s3}.json").write_text(
                json.dumps(ds_json, ensure_ascii=False, indent=2), encoding="utf-8")
            rich_json.unlink(missing_ok=True)
            srt_path = rec_out / f"{s3}.srt"
            if srt_path.exists():
                srt_path.unlink(missing_ok=True)
            if drive_root is not None:
                ddir = drive_root / rec / s3
                ddir.mkdir(parents=True, exist_ok=True)
                (ddir / f"{s3}.pb").write_bytes(pb_bytes)
            print(f"  [pb] {rec} {s3}: {len(pb_bytes)} bytes, "
                  f"{len(records)} words, {ds_json['metadata']['repeatEntries']} repeats")
            if args.upload and api is not None:
                api.upload_file(
                    path_or_fileobj=str(surah_dir / f"{s3}.pb"),
                    path_in_repo=f"{rec}/{s3}/{s3}.pb",
                    repo_id=args.repo_id, repo_type="dataset",
                )
                print(f"  [uploaded] {rec}/{s3}/{s3}.pb")

    print("\nDone. .pb files live under", str(out_root))


if __name__ == "__main__":
    main()
