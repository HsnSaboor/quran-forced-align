#!/usr/bin/env python3
"""End-to-End High-Throughput Colab Pipeline Runner for Quran Forced Alignment.

Orchestrates batch downloading, GPU loudness normalization, and CUDA forced alignment
across 225 reciters and 114 surahs from `verified_curl_cffi.json`.

All outputs (.opus audio and .json/.srt files) are written directly to Google Drive
with zero download/storage impact on your local machine.

Features:
  - Google Colab Auto-Detection & automatic Google Drive mount (`/content/drive`).
  - Persistent Google Drive model/token caching & local NVMe scratch synchronization.
  - Granular resume / incremental execution (verifies .json and .opus in Google Drive).
  - High-speed async audio download with curl_cffi / urllib connection retries.
  - Double-buffered producer-consumer pipeline with multithreaded FFmpeg EBU R128 loudnorm.
  - Batched CUDA inference + intra-surah silence splitting + segmented trellis (<0.15s).
  - Pure C fast repeat detection (<0.02s) + auto-Isti'adha preamble handling.
  - Live progress display and comprehensive Markdown summary reporting saved to Drive.

Usage Examples:
  # Run a single reciter by slug (e.g. mishary-rashid-alafasy):
  python scripts/colab_pipeline.py --reciter-slug mishary-rashid-alafasy --batch-size 8

  # Run a single reciter by index (0-indexed in verified_curl_cffi.json):
  python scripts/colab_pipeline.py --reciter-idx 0 --surahs 1-114

  # Run a range of reciters:
  python scripts/colab_pipeline.py --reciter-idx 0-5 --device cuda --batch-size 8

  # Run all 225 reciters with resume mode (skips existing outputs in Google Drive):
  python scripts/colab_pipeline.py --all-reciters --skip-existing --batch-size 8
"""

import argparse
import asyncio
import datetime
import glob
import json
import os
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["PYTHONUNBUFFERED"] = "1"

import shutil
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure local package source is in sys.path
_repo_src = str(Path(__file__).resolve().parent.parent / "src")
if _repo_src not in sys.path:
    sys.path.insert(0, _repo_src)

# Terminal coloring helpers
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RED = "\033[91m"
RESET = "\033[0m"


# ============================================================================
# Environment & Google Drive Management
# ============================================================================

def is_colab_environment() -> bool:
    """Detect if running inside Google Colab."""
    return "google.colab" in sys.modules or os.path.exists("/content")


def ensure_google_drive_mounted(gdrive_dir: str) -> None:
    """Detect and mount Google Drive if running in Colab."""
    if is_colab_environment() and gdrive_dir.startswith("/content/drive"):
        if not os.path.exists("/content/drive/MyDrive"):
            print(f"{CYAN}[Colab Setup]{RESET} Mounting Google Drive at /content/drive...")
            try:
                from google.colab import drive
                drive.mount("/content/drive", force_remount=False)
                print(f"{GREEN}✓ Google Drive successfully mounted.{RESET}")
            except Exception as e:
                print(f"{YELLOW}⚠️ Could not mount Google Drive automatically: {e}{RESET}")
                print("  Proceeding with local storage fallback if directory exists.")


def setup_directory_structure(gdrive_dir: str, local_cache_dir: str) -> Dict[str, str]:
    """Create persistent Google Drive directories and fast local NVMe cache paths."""
    paths = {
        "gdrive_root": gdrive_dir,
        "gdrive_json": os.path.join(gdrive_dir, "json"),
        "gdrive_opus": os.path.join(gdrive_dir, "opus"),
        "gdrive_models": os.path.join(gdrive_dir, "models"),
        "gdrive_reports": os.path.join(gdrive_dir, "reports"),
        "local_root": local_cache_dir,
        "local_audio": os.path.join(local_cache_dir, "audio"),
        "local_models": os.path.join(local_cache_dir, "models"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def sync_model_and_tokens(
    paths: Dict[str, str],
    device: str = "cuda",
    custom_model_path: Optional[str] = None,
    custom_tokens_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Ensure Zipformer ONNX model and tokens.txt are ready in fast NVMe cache and backed up to Drive."""
    from quran_forced_align.model_manager import (
        DEFAULT_FP16_MODEL,
        DEFAULT_INT8_MODEL,
        DEFAULT_TOKENS,
        resolve_model,
        resolve_tokens,
    )

    preferred_model_name = DEFAULT_FP16_MODEL if device == "cuda" else DEFAULT_INT8_MODEL
    tokens_name = DEFAULT_TOKENS

    local_model = os.path.join(paths["local_models"], preferred_model_name)
    local_tokens = os.path.join(paths["local_models"], tokens_name)
    drive_model = os.path.join(paths["gdrive_models"], preferred_model_name)
    drive_tokens = os.path.join(paths["gdrive_models"], tokens_name)

    # 1. Check custom overrides
    if custom_model_path and os.path.exists(custom_model_path):
        local_model = custom_model_path
    if custom_tokens_path and os.path.exists(custom_tokens_path):
        local_tokens = custom_tokens_path

    # 2. Check Google Drive cache
    if not os.path.exists(local_model) and os.path.exists(drive_model):
        print(f"{CYAN}[Model Setup]{RESET} Restoring model from Google Drive cache ({drive_model})...")
        shutil.copy2(drive_model, local_model)
    if not os.path.exists(local_tokens) and os.path.exists(drive_tokens):
        print(f"{CYAN}[Model Setup]{RESET} Restoring tokens from Google Drive cache ({drive_tokens})...")
        shutil.copy2(drive_tokens, local_tokens)

    # 3. Auto-resolve/download via model_manager if still missing
    if not os.path.exists(local_model):
        print(f"{CYAN}[Model Setup]{RESET} Resolving model from Hugging Face...")
        resolved_m = resolve_model(device=device, prefer_fp16=(device == "cuda"))
        shutil.copy2(resolved_m, local_model)
        try:
            shutil.copy2(local_model, drive_model)
            print(f"{GREEN}✓ Model backed up to Google Drive: {drive_model}{RESET}")
        except Exception:
            pass

    if not os.path.exists(local_tokens):
        print(f"{CYAN}[Model Setup]{RESET} Resolving tokens.txt from Hugging Face...")
        resolved_t = resolve_tokens()
        shutil.copy2(resolved_t, local_tokens)
        try:
            shutil.copy2(local_tokens, drive_tokens)
            print(f"{GREEN}✓ Tokens backed up to Google Drive: {drive_tokens}{RESET}")
        except Exception:
            pass

    print(
        f"{GREEN}✓ Model ready:{RESET} {local_model} ({os.path.getsize(local_model) / 1024 / 1024:.1f} MB)"
    )
    return local_model, local_tokens


# ============================================================================
# Reciter Selection & Surah Filtering
# ============================================================================

def parse_surah_spec(spec: str) -> List[int]:
    """Parse surah list string e.g. '1-114', '67-71', '1,2,3'."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        start_s, _, end_s = spec.partition("-")
        start, end = int(start_s), int(end_s)
        if end < start:
            raise ValueError(f"Invalid surah range: {spec}")
        return list(range(start, end + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def parse_reciter_idx_spec(spec: str) -> List[int]:
    """Parse reciter index range or list e.g. '0', '0-5', '0,1,2'."""
    spec = str(spec).strip()
    if "-" in spec and "," not in spec:
        s_s, _, e_s = spec.partition("-")
        s, e = int(s_s), int(e_s)
        return list(range(s, e + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def load_verified_dataset(json_path: str) -> List[Dict[str, Any]]:
    """Load and validate verified_curl_cffi.json."""
    if not os.path.exists(json_path):
        candidates = [
            json_path,
            "verified_curl_cffi.json",
            os.path.join(os.path.dirname(__file__), "..", "verified_curl_cffi.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                json_path = c
                break
        else:
            raise FileNotFoundError(f"Could not locate verified_curl_cffi.json at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "verified" in data:
        return data["verified"]
    elif isinstance(data, list):
        return data
    raise ValueError(f"Unexpected structure in {json_path}")


def filter_reciters(
    reciters: List[Dict[str, Any]],
    reciter_idx_spec: Optional[str] = None,
    reciter_slug_spec: Optional[str] = None,
    all_reciters: bool = False,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Filter reciters list according to CLI options, returning (original_idx, reciter_data)."""
    if all_reciters:
        return list(enumerate(reciters))

    selected: List[Tuple[int, Dict[str, Any]]] = []

    if reciter_idx_spec is not None:
        indices = set(parse_reciter_idx_spec(reciter_idx_spec))
        for idx in sorted(indices):
            if 0 <= idx < len(reciters):
                selected.append((idx, reciters[idx]))
            else:
                print(f"{YELLOW}⚠️ Reciter index {idx} out of range (0-{len(reciters)-1}){RESET}")

    if reciter_slug_spec is not None:
        target_slugs = {s.strip().lower() for s in reciter_slug_spec.split(",") if s.strip()}
        for idx, r in enumerate(reciters):
            slug = r.get("slug", "").lower()
            name = r.get("name", "").lower()
            if slug in target_slugs or name in target_slugs:
                if (idx, r) not in selected:
                    selected.append((idx, r))

    if not selected and not all_reciters:
        print(f"{YELLOW}No reciter specified. Defaulting to first reciter (index 0).{RESET}")
        selected.append((0, reciters[0]))

    return selected


# ============================================================================
# Incremental Execution & Cache Verification
# ============================================================================

def check_completed_surahs(
    reciter_slug: str,
    target_surahs: List[int],
    gdrive_json_dir: str,
    gdrive_opus_dir: str,
) -> Tuple[List[int], List[int]]:
    """Check which surahs already have valid .json and .opus in Google Drive."""
    rec_json_dir = os.path.join(gdrive_json_dir, reciter_slug)
    rec_opus_dir = os.path.join(gdrive_opus_dir, reciter_slug)

    completed = []
    remaining = []

    for s in target_surahs:
        json_path = os.path.join(rec_json_dir, f"{s:03d}.json")
        opus_path = os.path.join(rec_opus_dir, f"{s:03d}.opus")

        json_ok = os.path.exists(json_path) and os.path.getsize(json_path) > 100
        opus_ok = os.path.exists(opus_path) and os.path.getsize(opus_path) > 1000

        if json_ok and opus_ok:
            completed.append(s)
        else:
            remaining.append(s)

    return completed, remaining


# ============================================================================
# Asynchronous Audio Prefetcher (curl_cffi)
# ============================================================================

async def download_surahs_async(
    reciter_data: Dict[str, Any],
    surahs_to_download: List[int],
    out_dir: str,
    max_concurrent: int = 8,
) -> int:
    """Download required surah MP3s concurrently using curl_cffi with backoff retries."""
    if not surahs_to_download:
        return 0

    os.makedirs(out_dir, exist_ok=True)
    surahs_map = reciter_data.get("surahs", {})

    try:
        from curl_cffi.requests import AsyncSession
        has_curl_cffi = True
    except ImportError:
        has_curl_cffi = False

    sem = asyncio.Semaphore(max_concurrent)
    success_count = 0

    async def fetch_one(s_num: int) -> bool:
        nonlocal success_count
        s_str = str(s_num)
        target_path = os.path.join(out_dir, f"{s_num:03d}.mp3")

        # Check existing valid audio file
        if os.path.exists(target_path) and os.path.getsize(target_path) > 10000:
            success_count += 1
            return True

        s_info = surahs_map.get(s_str)
        if not s_info or not isinstance(s_info, dict):
            return False

        mp3_url = s_info.get("mp3_url")
        if not mp3_url:
            return False

        async with sem:
            for attempt in range(3):
                try:
                    if has_curl_cffi:
                        async with AsyncSession(timeout=60, impersonate="chrome120") as session:
                            resp = await session.get(mp3_url)
                            if resp.status_code == 200 and len(resp.content) > 10000:
                                with open(target_path, "wb") as f:
                                    f.write(resp.content)
                                success_count += 1
                                return True
                    else:
                        import urllib.request
                        req = urllib.request.Request(mp3_url, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(req, timeout=60) as resp:
                            data = resp.read()
                            if len(data) > 10000:
                                with open(target_path, "wb") as f:
                                    f.write(data)
                                success_count += 1
                                return True
                except Exception:
                    await asyncio.sleep(1.0 * (attempt + 1))
        return False

    tasks = [fetch_one(s) for s in surahs_to_download]
    await asyncio.gather(*tasks)
    return success_count


def _safe_run_async(coro):
    """Run an async coroutine safely whether an event loop is already running (e.g. IPython/Colab) or not."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


# ============================================================================
# Progress Display & Terminal Reporting
# ============================================================================

def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def generate_markdown_report(
    summaries: List[Dict[str, Any]],
    total_wall_sec: float,
    out_path: str,
) -> str:
    """Generate GitHub Flavored Markdown summary report of the alignment run."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_surahs_succeeded = sum(s.get("surahs_succeeded", 0) for s in summaries)
    total_surahs_failed = sum(s.get("surahs_failed", 0) for s in summaries)
    total_audio_hours = sum(s.get("audio_hours", 0.0) for s in summaries)
    total_words = sum(s.get("words", 0) for s in summaries)
    total_repeats = sum(s.get("repeats", 0) for s in summaries)

    overall_rtf = (total_wall_sec / (total_audio_hours * 3600.0)) if total_audio_hours > 0 else 0.0
    overall_speedup = (1.0 / overall_rtf) if overall_rtf > 0 else 0.0

    lines = [
        f"# Quran Forced Alignment — Pipeline Summary Report",
        f"",
        f"**Run Date:** `{timestamp}`  ",
        f"**Reciters Processed:** `{len(summaries)}`  ",
        f"**Total Surahs Aligned:** `{total_surahs_succeeded}` (Failed: `{total_surahs_failed}`)  ",
        f"**Total Audio Aligned:** `{total_audio_hours:.2f} hours`  ",
        f"**Total Wall Time:** `{format_time(total_wall_sec)}`  ",
        f"**Overall Pipeline Speedup:** `{overall_speedup:.1f}x realtime` (RTF: `{overall_rtf:.4f}x`)  ",
        f"**Total Words Aligned:** `{total_words:,}` | **Repeats Fixed:** `{total_repeats:,}`  ",
        f"",
        f"---",
        f"",
        f"## Reciter Breakdown",
        f"",
        f"| # | Reciter Slug | Surahs | Audio (hrs) | Wall Time | RTF | Speedup | Words | Repeats | Status |",
        f"|---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for idx, s in enumerate(summaries, 1):
        slug = s["reciter"]
        surahs_str = f"{s['surahs_succeeded']}/{s.get('total_requested', 114)}"
        audio_h = f"{s.get('audio_hours', 0.0):.2f}"
        wall_str = format_time(s.get("wall_sec", 0.0))
        rtf_val = s.get("rtf", 0.0)
        rtf_str = f"{rtf_val:.4f}x" if rtf_val > 0 else "-"
        speedup_str = f"{1.0/max(rtf_val, 1e-6):.1f}x" if rtf_val > 0 else "-"
        words_cnt = f"{s.get('words', 0):,}"
        repeats_cnt = f"{s.get('repeats', 0):,}"
        status_tag = f"**`{s['status']}`**"

        lines.append(
            f"| {idx} | `{slug}` | {surahs_str} | {audio_h} | {wall_str} | {rtf_str} | {speedup_str} | {words_cnt} | {repeats_cnt} | {status_tag} |"
        )

    lines.extend([
        f"",
        f"---",
        f"Generated automatically by `scripts/colab_pipeline.py`.",
    ])

    report_content = "\n".join(lines)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    return report_content


# ============================================================================
# Main Pipeline Runner
# ============================================================================

def run_pipeline(args: argparse.Namespace) -> None:
    """Main execution entry point."""
    pipeline_t0 = time.monotonic()

    print(f"\n{BOLD}{CYAN}{'='*80}{RESET}")
    print(f"{BOLD}{CYAN}  QURAN FORCED ALIGNMENT — END-TO-END COLAB PIPELINE RUNNER{RESET}")
    print(f"{BOLD}{CYAN}{'='*80}{RESET}\n")

    # 1. Colab & Google Drive Mount Detection
    ensure_google_drive_mounted(args.gdrive_dir)
    paths = setup_directory_structure(args.gdrive_dir, args.local_cache_dir)

    # 2. Device & Hardware Auto-Detection
    from quran_forced_align.model_manager import resolve_device, verify_colab_environment

    env_status = verify_colab_environment()
    resolved_device = resolve_device(args.device)
    print(f"• Runtime Environment : {'Google Colab' if env_status['is_colab'] else 'Local / Server'}")
    print(f"• Active Device        : {resolved_device.upper()} ({env_status.get('gpu_name') or 'CPU'})")
    print(f"• CUDA Provider Linked : {env_status.get('ort_gpu_ok')}")
    print(f"• Google Drive Storage : {paths['gdrive_root']}")
    print(f"• Local Scratch NVMe   : {paths['local_root']}")

    # 3. Model & Token Auto-Resolution
    local_model, local_tokens = sync_model_and_tokens(
        paths,
        device=resolved_device,
        custom_model_path=args.model_path,
        custom_tokens_path=args.tokens_path,
    )

    # 4. Load & Filter Reciters
    all_reciters_data = load_verified_dataset(args.verified_json)
    target_reciters = filter_reciters(
        all_reciters_data,
        reciter_idx_spec=args.reciter_idx,
        reciter_slug_spec=args.reciter_slug,
        all_reciters=args.all_reciters,
    )
    target_surahs = parse_surah_spec(args.surahs)

    print(f"\n{BOLD}Execution Plan:{RESET}")
    print(f"• Reciters Selected   : {len(target_reciters)} / {len(all_reciters_data)}")
    print(f"• Surahs per Reciter  : {len(target_surahs)} surahs ({target_surahs[0]} to {target_surahs[-1]})")
    print(f"• Batch Size (CUDA)   : {args.batch_size}")
    print(f"• Skip Existing Runs  : {args.skip_existing}")
    print(f"• Transcode Opus/EBU  : True (EBU R128 loudnorm enabled)")
    print(f"{'-'*80}\n")

    # Import alignment engine
    from quran_forced_align.batch_cli import run_pipelined_batch

    reciter_summaries: List[Dict[str, Any]] = []

    for r_idx, (orig_idx, r_data) in enumerate(target_reciters, 1):
        slug = r_data.get("slug", f"reciter_{orig_idx}")
        rec_name = r_data.get("name", slug)

        rec_drive_json = os.path.join(paths["gdrive_json"], slug)
        rec_drive_opus = os.path.join(paths["gdrive_opus"], slug)
        rec_local_audio = os.path.join(paths["local_audio"], slug)

        os.makedirs(rec_drive_json, exist_ok=True)
        os.makedirs(rec_drive_opus, exist_ok=True)
        os.makedirs(rec_local_audio, exist_ok=True)

        print(f"\n{BOLD}[{r_idx}/{len(target_reciters)}] Reciter: {slug} ({rec_name}) [Idx: {orig_idx}]{RESET}")

        # Check incremental resume
        if args.skip_existing:
            completed_surahs, remaining_surahs = check_completed_surahs(
                slug, target_surahs, paths["gdrive_json"], paths["gdrive_opus"]
            )
            if len(remaining_surahs) == 0:
                print(f"  {GREEN}✓ [SKIP] All {len(target_surahs)} surahs already aligned in Google Drive.{RESET}")
                reciter_summaries.append({
                    "reciter": slug,
                    "status": "already_completed",
                    "total_requested": len(target_surahs),
                    "surahs_succeeded": len(target_surahs),
                    "surahs_failed": 0,
                    "audio_hours": 0.0,
                    "wall_sec": 0.0,
                    "rtf": 0.0,
                    "words": 0,
                    "repeats": 0,
                })
                continue
            elif len(completed_surahs) > 0:
                print(f"  {CYAN}⚡ Resuming: {len(completed_surahs)} surahs exist, {len(remaining_surahs)} remaining to align.{RESET}")
                active_surahs = remaining_surahs
            else:
                active_surahs = target_surahs
        else:
            active_surahs = target_surahs

        # Step 1: Parallel Download Audio to local NVMe
        print(f"  [1/3] Downloading {len(active_surahs)} surahs to local scratch NVMe ({rec_local_audio})...")
        t0_dl = time.monotonic()
        dl_count = _safe_run_async(
            download_surahs_async(
                r_data,
                active_surahs,
                rec_local_audio,
                max_concurrent=args.max_concurrent_dl,
            )
        )
        dl_sec = time.monotonic() - t0_dl
        print(f"        {GREEN}✓ {dl_count}/{len(active_surahs)} audio files downloaded ({dl_sec:.1f}s).{RESET}")

        # Check available local files
        available_files = sorted([
            int(os.path.basename(f).split(".")[0])
            for f in glob.glob(f"{rec_local_audio}/*.mp3")
            if int(os.path.basename(f).split(".")[0]) in active_surahs
        ])

        if not available_files:
            print(f"  {RED}⚠ No audio available for {slug}, skipping alignment.{RESET}")
            reciter_summaries.append({
                "reciter": slug,
                "status": "failed_download",
                "total_requested": len(target_surahs),
                "surahs_succeeded": 0,
                "surahs_failed": len(active_surahs),
                "audio_hours": 0.0,
                "wall_sec": 0.0,
                "rtf": 0.0,
                "words": 0,
                "repeats": 0,
            })
            continue

        # Step 2: Overlapped GPU Forced Alignment + Loudnorm Opus Transcode
        print(f"  [2/3] Running Pipelined GPU Alignment & Loudnorm Transcoding...")
        t0_align = time.monotonic()

        batch_result = run_pipelined_batch(
            surah_list=available_files,
            audio_dir=rec_local_audio,
            out_dir=rec_drive_json,
            opus_dir=rec_drive_opus if args.transcode_opus else None,
            transcode_opus=args.transcode_opus,
            model_path=local_model,
            tokens_path=local_tokens,
            device=resolved_device,
            cuda_batch_size=args.batch_size if resolved_device == "cuda" else 1,
            intra_surah_split=True,
            prefetch_workers=args.prefetch_workers,
            prefetch_batches=args.prefetch_batches,
            verbose=True,
        )

        align_wall_sec = time.monotonic() - t0_align
        audio_hrs = batch_result["total_audio_sec"] / 3600.0
        rtf = batch_result["overall_rtf"]
        speedup = 1.0 / max(rtf, 1e-6)

        # Step 3: Cleanup local scratch audio cache to prevent Colab disk exhaustion
        if args.clean_local_audio:
            shutil.rmtree(rec_local_audio, ignore_errors=True)
            os.makedirs(rec_local_audio, exist_ok=True)

        succeeded = batch_result["succeeded_count"]
        failed = batch_result["failed_count"]
        status_str = "ok" if failed == 0 else ("partial" if succeeded > 0 else "failed")

        print(
            f"  {GREEN}✓ [3/3] Finished {slug}: {succeeded}/{len(available_files)} surahs aligned in {format_time(align_wall_sec)} "
            f"({audio_hrs:.2f}h audio | RTF: {rtf:.4f}x | {speedup:.1f}x realtime){RESET}"
        )

        reciter_summaries.append({
            "reciter": slug,
            "status": status_str,
            "total_requested": len(target_surahs),
            "surahs_succeeded": succeeded + (len(target_surahs) - len(active_surahs)),
            "surahs_failed": failed,
            "audio_hours": audio_hrs,
            "wall_sec": align_wall_sec,
            "rtf": rtf,
            "words": batch_result["total_words"],
            "repeats": batch_result["total_repeats"],
        })

    # ========================================================================
    # Summary Report Generation
    # ========================================================================
    total_pipeline_sec = time.monotonic() - pipeline_t0
    report_filename = f"alignment_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = os.path.join(paths["gdrive_reports"], report_filename)

    md_report = generate_markdown_report(reciter_summaries, total_pipeline_sec, report_path)

    print(f"\n\n{BOLD}{GREEN}{'='*80}{RESET}")
    print(f"{BOLD}{GREEN}  PIPELINE EXECUTION COMPLETE!{RESET}")
    print(f"{BOLD}{GREEN}{'='*80}{RESET}\n")
    print(md_report)
    print(f"\n{CYAN}✓ Summary Markdown saved to: {report_path}{RESET}\n")


# ============================================================================
# CLI Argument Parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-End Colab Pipeline Runner for Quran Forced Alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Reciter selection flags
    rec_group = parser.add_argument_group("Reciter Selection")
    rec_group.add_argument(
        "--reciter-idx",
        type=str,
        default=None,
        help="0-indexed reciter index or range (e.g. '0', '0-5', '0,1,2')",
    )
    rec_group.add_argument(
        "--reciter-slug",
        type=str,
        default=None,
        help="Reciter slug or comma-separated slugs (e.g. 'mishary-rashid-alafasy', 'sudais')",
    )
    rec_group.add_argument(
        "--all-reciters",
        action="store_true",
        help="Process all reciters in verified_curl_cffi.json",
    )

    # Surah and input dataset flags
    data_group = parser.add_argument_group("Surah & Input Options")
    data_group.add_argument(
        "--verified-json",
        type=str,
        default="verified_curl_cffi.json",
        help="Path to verified reciters JSON file",
    )
    data_group.add_argument(
        "--surahs",
        type=str,
        default="1-114",
        help="Surah range or comma-separated list (e.g. '1-114', '67-71', '1,2,3')",
    )

    # Storage & directories
    storage_group = parser.add_argument_group("Storage Directories")
    storage_group.add_argument(
        "--gdrive-dir",
        type=str,
        default="/content/drive/MyDrive/QuranAligned",
        help="Root Google Drive directory for persistent outputs & model cache",
    )
    storage_group.add_argument(
        "--local-cache-dir",
        type=str,
        default="/content/quran_scratch",
        help="Fast local NVMe directory for scratch audio decoding",
    )

    # Hardware & batch tuning flags
    hw_group = parser.add_argument_group("Hardware & Performance Options")
    hw_group.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device",
    )
    hw_group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for CUDA inference",
    )
    hw_group.add_argument(
        "--trt",
        action="store_true",
        default=True,
        help="Enable TensorRT Execution Provider if available",
    )
    hw_group.add_argument(
        "--no-trt",
        action="store_false",
        dest="trt",
        help="Disable TensorRT Execution Provider",
    )
    hw_group.add_argument(
        "--prefetch-workers",
        type=int,
        default=4,
        help="Background CPU threads for audio decoding & Fbank feature extraction",
    )
    hw_group.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="Queue depth for double-buffering prefetch",
    )
    hw_group.add_argument(
        "--max-concurrent-dl",
        type=int,
        default=8,
        help="Maximum concurrent audio downloads per reciter",
    )

    # Resume and maintenance flags
    misc_group = parser.add_argument_group("Execution Control")
    misc_group.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip surahs that already have .json and .opus in Google Drive",
    )
    misc_group.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Re-process all surahs even if output files already exist",
    )
    misc_group.add_argument(
        "--transcode-opus",
        action="store_true",
        default=True,
        help="Transcode audio to 96kbps Opus with EBU R128 loudness normalization",
    )
    misc_group.add_argument(
        "--no-transcode-opus",
        action="store_false",
        dest="transcode_opus",
        help="Disable Opus transcoding (Pure alignment only: 225x+ realtime)",
    )
    misc_group.add_argument(
        "--clean-local-audio",
        action="store_true",
        default=True,
        help="Remove local MP3 files after processing each reciter to preserve disk space",
    )
    misc_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional override path to ONNX model",
    )
    misc_group.add_argument(
        "--tokens-path",
        type=str,
        default=None,
        help="Optional override path to tokens.txt",
    )

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_pipeline(cli_args)
