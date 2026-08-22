"""Batch CLI: high-throughput, overlapped asynchronous forced-alignment pipeline
for multiple surahs with double-buffered producer-consumer execution.

  quran-forced-align-batch --surahs 1-114 --audio-dir audio --out-dir output/json --device cuda --cuda-batch-size 8 --intra-surah-split

WORKER PIPELINE STAGES & OVERLAPPED EXECUTION:
----------------------------------------------
  Stage 1: Async Audio Cache Prefetch / Resolution (finds .mp3, .opus, .wav, etc.)
  Stage 2: Multithreaded Audio Decode & Loudnorm + Opus Transcoding (ffmpeg)
  Stage 3: Parallel Feature Extraction (Fbank) + Reference Token Compilation + Silence Midpoints
  Stage 4: GPU Batched Inference & CUDA Forced Alignment (persistent CUDAEngine + intra-surah split)
  Stage 5: JSON/SRT Rich Export & Output Synchronization

By double-buffering Stages 1-3 with Stage 4 in a bounded producer-consumer queue:
  - While the GPU computes batched inference on batch K, background CPU worker threads
    are already decoding audio and extracting Fbank features for batch K+1.
  - The ONNX Runtime CUDA session and GPU context are initialized ONCE and remain resident,
    eliminating per-batch context creation overhead.
  - GPU inference is NEVER blocked waiting on disk I/O, network downloads, or ffmpeg decoding.
"""
import argparse
import os
import queue
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .audio import find_audio_file, load_audio_as_wav16k, transcode_to_opus
from .cli import add_tuning_args
from .constants import (
    DEFAULT_ANOMALY_HIGH_RATIO,
    DEFAULT_ANOMALY_LOW_RATIO,
    DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
    DEFAULT_MAX_REPEAT_WINDOW_WORDS,
    DEFAULT_REPEAT_CONFIDENCE_MARGIN,
    DEFAULT_TAIL_SILENCE_SEC,
    FBANK_FRAME_SHIFT_SAMPLES,
    SAMPLE_RATE,
)
from .engines import get_engine
from .features import compute_fbank_features
from .pipeline import _align_from_log_probs, _build_surah_inputs, align_surah, align_surahs_batched
from .reference import build_combined_reference
from .silence import find_silence_midpoints
from .srt import emit_json_rich, emit_srt
from .tokenizer import load_tokens


def parse_surah_list(spec: str) -> list[int]:
    """Parse --surahs, accepting either a range ("67-71") or a comma list
    ("67,68,69")."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo_s, _, hi_s = spec.partition("-")
        lo, hi = int(lo_s), int(hi_s)
        if hi < lo:
            raise ValueError(f"invalid --surahs range {spec!r}: end < start")
        return list(range(lo, hi + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def _chunked(items, size):
    """Split `items` into consecutive sub-lists of at most `size` each --
    the batching granularity `--cuda-batch-size` controls."""
    return [items[i:i + size] for i in range(0, len(items), size)]


@dataclass
class PreparedSurah:
    """Pre-processed surah ready for GPU inference and alignment."""
    surah: int
    audio_path: str
    tokens_tuple: tuple
    combined_token_ids: list[int]
    word_slots: list
    feats: np.ndarray
    samples: np.ndarray
    silence_frames: list[int]
    audio_duration_sec: float
    opus_path: str | None = None
    error: Exception | None = None


@dataclass
class PreparedBatch:
    """A batch of pre-processed surahs ready for batched GPU execution."""
    batch_idx: int
    surahs: list[int]
    items: list[PreparedSurah]
    prepare_wall_sec: float = 0.0
    error: Exception | None = None


def _prepare_single_surah_worker(
    surah: int,
    audio_dir: str,
    tokens_info: tuple,
    tail_silence_sec: float,
    intra_surah_split: bool,
    opus_dir: str | None,
    transcode_opus_flag: bool,
    include_istiaatha: bool = True,
) -> PreparedSurah:
    """Worker task: loads audio, extracts Fbank features, computes reference tokens,
    detects silence split points, and optionally transcodes to Opus with loudnorm.
    """
    audio_path = find_audio_file(audio_dir, surah)
    if audio_path is None:
        candidate = os.path.join(audio_dir, f"{surah:03d}.mp3")
        if os.path.exists(candidate):
            audio_path = candidate
        else:
            raise FileNotFoundError(f"audio file for surah {surah:03d} not found in {audio_dir}")

    tok2id, id2tok, blank_id, max_token_len = tokens_info
    combined_token_ids, word_slots = build_combined_reference(surah, tok2id, max_token_len, include_istiaatha=include_istiaatha)

    # Decode audio to 16kHz mono PCM
    samples = load_audio_as_wav16k(audio_path, threads=2)
    audio_duration_sec = len(samples) / SAMPLE_RATE

    # Extract deterministic Fbank features
    feats = compute_fbank_features(samples, tail_silence_sec=tail_silence_sec)

    # Silence midpoint detection for intra-surah split
    silence_frames: list[int] = []
    if intra_surah_split:
        silence_midpoints = find_silence_midpoints(samples, SAMPLE_RATE)
        silence_frames = [pos // FBANK_FRAME_SHIFT_SAMPLES for pos in silence_midpoints]

    # Optional Opus transcoding with loudnorm
    opus_path = None
    if transcode_opus_flag and opus_dir:
        os.makedirs(opus_dir, exist_ok=True)
        dest_opus = os.path.join(opus_dir, f"{surah:03d}.opus")
        transcode_to_opus(audio_path, dest_opus, loudnorm=True, threads=2)
        opus_path = dest_opus

    return PreparedSurah(
        surah=surah,
        audio_path=audio_path,
        tokens_tuple=tokens_info,
        combined_token_ids=combined_token_ids,
        word_slots=word_slots,
        feats=feats,
        samples=samples,
        silence_frames=silence_frames,
        audio_duration_sec=audio_duration_sec,
        opus_path=opus_path,
    )


def run_pipelined_batch(
    surah_list: list[int],
    audio_dir: str,
    out_dir: str,
    *,
    model_path: str = "model/zipformer_p_arabic_v3.int8.onnx",
    tokens_path: str = "model/tokens.txt",
    device: str = "cuda",
    cuda_batch_size: int = 8,
    intra_surah_split: bool = True,
    opus_dir: str | None = None,
    transcode_opus: bool = False,
    include_istiaatha: bool = True,
    prefetch_workers: int = 4,
    prefetch_batches: int = 2,
    anomaly_low_ratio: float = DEFAULT_ANOMALY_LOW_RATIO,
    anomaly_high_ratio: float = DEFAULT_ANOMALY_HIGH_RATIO,
    ayah_final_high_ratio_mult: float = DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
    repeat_confidence_margin: float = DEFAULT_REPEAT_CONFIDENCE_MARGIN,
    max_repeat_window_words: int | None = DEFAULT_MAX_REPEAT_WINDOW_WORDS,
    tail_silence_sec: float = DEFAULT_TAIL_SILENCE_SEC,
    verbose: bool = True,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Run the high-throughput, overlapped asynchronous forced-alignment pipeline.

    Pipelining details:
      - Double-buffered Queue: Producer thread runs parallel audio loading,
        loudnorm Opus transcoding, and Fbank feature extraction ahead of time.
      - Persistent Engine: The ONNX Runtime CUDA session and GPU context are
        initialized ONCE, eliminating re-initialization overhead.
      - Batched CUDA Inference + Intra-Surah Split: Maximize GPU Tensor/CUDA core
        occupancy across multiple surahs and internal silence segments simultaneously.
      - Non-blocking Output Export: Output files (.srt and .json) are emitted
        immediately upon batch completion.
    """
    os.makedirs(out_dir, exist_ok=True)
    if opus_dir:
        os.makedirs(opus_dir, exist_ok=True)
        transcode_opus = True

    t_start = time.monotonic()
    tokens_info = load_tokens(tokens_path)
    batches = _chunked(surah_list, cuda_batch_size if device == "cuda" else 1)

    batch_queue: queue.Queue[PreparedBatch | None] = queue.Queue(maxsize=max(1, prefetch_batches))
    producer_exception: list[Exception] = []

    def producer_loop():
        """Background producer thread: prepares batches of Fbank features & references."""
        try:
            with ThreadPoolExecutor(max_workers=max(1, prefetch_workers)) as executor:
                for batch_idx, batch_surahs in enumerate(batches):
                    t0_prep = time.monotonic()
                    future_to_surah = {
                        executor.submit(
                            _prepare_single_surah_worker,
                            s,
                            audio_dir,
                            tokens_info,
                            tail_silence_sec,
                            intra_surah_split if device == "cuda" else False,
                            opus_dir,
                            transcode_opus,
                            include_istiaatha,
                        ): s
                        for s in batch_surahs
                    }
                    items: list[PreparedSurah] = []
                    for future in as_completed(future_to_surah):
                        s = future_to_surah[future]
                        try:
                            item = future.result()
                            items.append(item)
                        except Exception as e:
                            err_item = PreparedSurah(
                                surah=s,
                                audio_path="",
                                tokens_tuple=tokens_info,
                                combined_token_ids=[],
                                word_slots=[],
                                feats=np.zeros((1, 80), dtype=np.float32),
                                samples=np.zeros(1, dtype=np.float32),
                                silence_frames=[],
                                audio_duration_sec=0.0,
                                error=e,
                            )
                            items.append(err_item)

                    items.sort(key=lambda it: batch_surahs.index(it.surah))
                    prep_wall = time.monotonic() - t0_prep
                    prep_batch = PreparedBatch(
                        batch_idx=batch_idx,
                        surahs=batch_surahs,
                        items=items,
                        prepare_wall_sec=prep_wall,
                    )
                    batch_queue.put(prep_batch)
        except Exception as e:
            producer_exception.append(e)
        finally:
            batch_queue.put(None)

    prod_thread = threading.Thread(target=producer_loop, name="qfa-prefetch-producer", daemon=True)
    prod_thread.start()

    if verbose:
        print(f"[pipeline] Initializing {device.upper()} engine ({model_path})...")
    engine = get_engine(device)(model_path)

    results: dict[int, dict] = {}
    errors: dict[int, Exception] = {}
    total_audio_sec = 0.0
    total_words = 0
    total_repeats = 0

    null_log = lambda msg: None

    while True:
        prepared = batch_queue.get()
        if prepared is None:
            break

        batch_surahs = prepared.surahs
        items = prepared.items
        t_batch_start = time.monotonic()

        valid_items = [it for it in items if it.error is None]
        for it in items:
            if it.error is not None:
                errors[it.surah] = it.error
                if verbose:
                    print(f"[surah {it.surah:03d}] PREP FAILED: {type(it.error).__name__}: {it.error}")

        if not valid_items:
            continue

        valid_surahs = [it.surah for it in valid_items]
        feats_list = [it.feats for it in valid_items]
        silence_frames_list = [it.silence_frames for it in valid_items]

        try:
            if device == "cuda" and len(valid_items) > 1:
                if intra_surah_split and hasattr(engine, "run_inference_batched_with_intra_surah_split"):
                    log_probs_list, seconds_per_frame = engine.run_inference_batched_with_intra_surah_split(
                        feats_list, silence_frames_list
                    )
                else:
                    log_probs_list, seconds_per_frame = engine.run_inference_batched(feats_list)
            elif device == "cuda" and len(valid_items) == 1:
                if intra_surah_split and hasattr(engine, "run_inference_intra_surah_split"):
                    log_probs, seconds_per_frame = engine.run_inference_intra_surah_split(
                        feats_list[0], silence_frames_list[0]
                    )
                else:
                    log_probs, seconds_per_frame = engine.run_inference(feats_list[0])
                log_probs_list = [log_probs]
            else:
                log_probs_list = []
                for feats in feats_list:
                    lp, spf = engine.run_inference(feats)
                    log_probs_list.append(lp)
                seconds_per_frame = spf

            batch_elapsed = time.monotonic() - t_batch_start

            for it, log_probs in zip(valid_items, log_probs_list):
                s = it.surah
                tok2id, id2tok, blank_id, max_token_len = it.tokens_tuple
                
                if hasattr(engine, "_last_log_probs_cpu"):
                    log_probs_copy = log_probs.copy()
                    engine._last_log_probs_cpu = log_probs_copy
                    engine._last_log_probs_gpu = engine._torch.as_tensor(
                        log_probs_copy, dtype=engine._torch.float32, device=engine._device
                    )
                    log_probs = log_probs_copy

                try:
                    if str(include_istiaatha).lower() in ("auto", "none"):
                        from .pipeline import detect_leading_istiaatha
                        has_istiaatha = detect_leading_istiaatha(log_probs, id2tok)
                        comb_ids, w_slots = build_combined_reference(s, tok2id, max_token_len, include_istiaatha=has_istiaatha)
                        strip_aya0 = True
                    else:
                        comb_ids = it.combined_token_ids
                        w_slots = it.word_slots
                        strip_aya0 = False

                    records = _align_from_log_probs(
                        engine,
                        log_probs,
                        seconds_per_frame,
                        comb_ids,
                        blank_id,
                        w_slots,
                        id2tok,
                        anomaly_low_ratio,
                        anomaly_high_ratio,
                        ayah_final_high_ratio_mult,
                        repeat_confidence_margin,
                        max_repeat_window_words,
                        null_log,
                        silence_feature_frames=it.silence_frames,
                        strip_istiaatha=strip_aya0,
                    )

                    out_path = os.path.join(out_dir, f"{s:03d}.srt")
                    emit_srt(records, out_path)
                    json_out = os.path.splitext(out_path)[0] + ".json"
                    emit_json_rich(records, json_out)

                    n_repeats = sum(1 for r in records if r.get("is_repeat"))
                    n_words = len(records)
                    audio_dur = it.audio_duration_sec
                    total_audio_sec += audio_dur
                    total_words += n_words
                    total_repeats += n_repeats

                    per_surah_wall = batch_elapsed / len(valid_items)
                    rtf = per_surah_wall / max(audio_dur, 0.001)

                    res = {
                        "surah": s,
                        "n_words": n_words,
                        "n_repeats": n_repeats,
                        "audio_duration_sec": audio_dur,
                        "wall_clock_sec": per_surah_wall,
                        "rtf": rtf,
                        "status": "ok",
                        "srt_path": out_path,
                        "json_path": json_out,
                        "opus_path": it.opus_path,
                    }
                    results[s] = res
                    if progress_callback is not None:
                        progress_callback(res)
                except Exception as e_surah:
                    errors[s] = e_surah
                    print(f"  [surah {s:03d}] ALIGNMENT/EXPORT FAILED: {type(e_surah).__name__}: {e_surah}", flush=True)

            if verbose:
                b_words = sum(results[s]["n_words"] for s in valid_surahs if s in results)
                b_dur = sum(results[s]["audio_duration_sec"] for s in valid_surahs if s in results)
                b_rtf = batch_elapsed / max(b_dur, 0.001)
                print(
                    f"[batch {valid_surahs}] {len(valid_surahs)} surahs ({b_words} words, {b_dur:.1f}s audio) "
                    f"aligned in {batch_elapsed:.2f}s (RTF {b_rtf:.4f}x, {1.0/max(b_rtf, 1e-6):.1f}x realtime)"
                )

        except Exception as e:
            for s in valid_surahs:
                errors[s] = e
            print(f"  [batch {valid_surahs}] INFERENCE FAILED: {type(e).__name__}: {e}", flush=True)

    prod_thread.join()
    total_wall_sec = time.monotonic() - t_start
    overall_rtf = total_wall_sec / max(total_audio_sec, 0.001)

    return {
        "results": results,
        "errors": errors,
        "total_surahs": len(surah_list),
        "succeeded_count": len(results),
        "failed_count": len(errors),
        "total_words": total_words,
        "total_repeats": total_repeats,
        "total_audio_sec": total_audio_sec,
        "total_wall_sec": total_wall_sec,
        "overall_rtf": overall_rtf,
    }


def _align_one_surah(surah: int, audio_dir: str, out_dir: str, model_path: str, tokens_path: str,
                      device: str, intra_surah_split: bool,
                      anomaly_low_ratio: float, anomaly_high_ratio: float,
                      ayah_final_high_ratio_mult: float, repeat_confidence_margin: float,
                      max_repeat_window_words: int | None, tail_silence_sec: float) -> dict:
    """Module-level, picklable worker: aligns ONE surah and writes its SRT +
    JSON output files. Must not be a closure/lambda -- ProcessPoolExecutor
    pickles the callable + args to send to the worker process.
    """
    t0 = time.monotonic()
    audio_path = find_audio_file(audio_dir, surah) or os.path.join(audio_dir, f"{surah:03d}.mp3")
    out_path = os.path.join(out_dir, f"{surah:03d}.srt")

    records = align_surah(
        surah, audio_path,
        model_path=model_path,
        tokens_path=tokens_path,
        device=device,
        intra_surah_split=intra_surah_split,
        anomaly_low_ratio=anomaly_low_ratio,
        anomaly_high_ratio=anomaly_high_ratio,
        ayah_final_high_ratio_mult=ayah_final_high_ratio_mult,
        repeat_confidence_margin=repeat_confidence_margin,
        max_repeat_window_words=max_repeat_window_words,
        tail_silence_sec=tail_silence_sec,
        verbose=False,
    )

    os.makedirs(out_dir, exist_ok=True)
    emit_srt(records, out_path)
    json_out = os.path.splitext(out_path)[0] + ".json"
    emit_json_rich(records, json_out)

    n_repeats = sum(1 for r in records if r["is_repeat"])
    return {
        "surah": surah,
        "n_words": len(records),
        "n_repeats": n_repeats,
        "wall_clock_sec": time.monotonic() - t0,
    }


def _align_batch_of_surahs(surahs: list[int], audio_dir: str, out_dir: str, model_path: str, tokens_path: str,
                            intra_surah_split: bool,
                            anomaly_low_ratio: float, anomaly_high_ratio: float,
                            ayah_final_high_ratio_mult: float, repeat_confidence_margin: float,
                            max_repeat_window_words: int | None, tail_silence_sec: float) -> list[dict]:
    """Module-level, picklable worker: aligns a BATCH of surahs together
    through `pipeline.align_surahs_batched`'s single batched CUDA
    inference call, and writes each surah's own SRT + JSON output files.
    """
    t0 = time.monotonic()
    audio_paths = [find_audio_file(audio_dir, surah) or os.path.join(audio_dir, f"{surah:03d}.mp3") for surah in surahs]

    records_list = align_surahs_batched(
        surahs, audio_paths,
        model_path=model_path,
        tokens_path=tokens_path,
        intra_surah_split=intra_surah_split,
        anomaly_low_ratio=anomaly_low_ratio,
        anomaly_high_ratio=anomaly_high_ratio,
        ayah_final_high_ratio_mult=ayah_final_high_ratio_mult,
        repeat_confidence_margin=repeat_confidence_margin,
        max_repeat_window_words=max_repeat_window_words,
        tail_silence_sec=tail_silence_sec,
        verbose=False,
    )

    os.makedirs(out_dir, exist_ok=True)
    elapsed = time.monotonic() - t0
    results = []
    for surah, records in zip(surahs, records_list):
        out_path = os.path.join(out_dir, f"{surah:03d}.srt")
        emit_srt(records, out_path)
        json_out = os.path.splitext(out_path)[0] + ".json"
        emit_json_rich(records, json_out)
        n_repeats = sum(1 for r in records if r["is_repeat"])
        results.append({
            "surah": surah,
            "n_words": len(records),
            "n_repeats": n_repeats,
            "wall_clock_sec": elapsed,
        })
    return results


def build_parser() -> argparse.ArgumentParser:
    from .cli import parse_istiaatha_choice
    ap = argparse.ArgumentParser(
        prog="quran-forced-align-batch",
        description="High-throughput asynchronous forced-alignment batch pipeline for Quran surahs",
    )
    ap.add_argument("--surahs", required=True, help="surah numbers to process (e.g. '1-114' or '67,68,69')")
    ap.add_argument("--audio-dir", required=True, help="directory containing input audio files")
    ap.add_argument("--out-dir", required=True, help="directory where output SRT/JSON files will be written")
    ap.add_argument(
        "--opus-dir",
        default=None,
        help="optional directory for saving normalized Opus audio",
    )
    ap.add_argument(
        "--transcode-opus",
        action="store_true",
        help="enable EBU R128 loudnorm Opus transcoding alongside forced alignment",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="path to Zipformer CTC ONNX model (auto-resolved from HF/cache if omitted)",
    )
    ap.add_argument(
        "--tokens",
        default=None,
        help="path to tokens.txt vocabulary file (auto-resolved from HF/cache if omitted)",
    )
    ap.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="forced-alignment execution engine: 'cpu' (default) or 'cuda'",
    )
    ap.add_argument(
        "--batch-size", "--cuda-batch-size",
        dest="cuda_batch_size",
        type=int,
        default=1,
        help="--device cuda ONLY: surahs per batched acoustic-model inference pass (recommended: 8 on Colab T4)",
    )
    ap.add_argument(
        "--trt",
        action="store_true",
        default=False,
        help="enable TensorRT execution provider for FP16 acceleration on NVIDIA GPUs",
    )
    ap.add_argument(
        "--intra-surah-split",
        action="store_true",
        help="--device cuda ONLY: split each surah across silence points into warm-up-overlapped segments",
    )
    ap.add_argument(
        "--prefetch-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="number of parallel CPU threads for audio decode, loudnorm, and Fbank feature extraction",
    )
    ap.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="double-buffering queue depth for pipelined producer-consumer execution",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=os.cpu_count(),
        help="maximum worker concurrency limit",
    )
    ap.add_argument(
        "--include-istiaatha",
        type=parse_istiaatha_choice,
        default="auto",
        help="include Isti'adha preamble in reference: 'auto' (default), 'yes', or 'no'",
    )
    ap.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="suppress per-batch progress logs",
    )
    add_tuning_args(ap)
    return ap


def _validate_device_flags(args) -> None:
    """Cross-flag validation for --device/--cuda-batch-size/--intra-surah-split."""
    if args.cuda_batch_size != 1 and args.device != "cuda":
        raise SystemExit("--cuda-batch-size > 1 requires --device cuda")
    if args.cuda_batch_size < 1:
        raise SystemExit("--cuda-batch-size must be >= 1")
    if args.intra_surah_split and args.device != "cuda":
        raise SystemExit("--intra-surah-split requires --device cuda")


def print_summary_report(summary: dict, surah_list: list[int]) -> None:
    """Render a clean, formatted terminal summary report."""
    results = summary["results"]
    errors = summary["errors"]

    print()
    print("=" * 80)
    print(f"{'surah':>6}  {'audio(s)':>9}  {'words':>6}  {'repeats':>8}  {'wall(s)':>8}  {'RTF':>9}  status")
    print("-" * 80)

    for s in surah_list:
        if s in results:
            r = results[s]
            audio_dur = r.get("audio_duration_sec", 0.0)
            wall_sec = r.get("wall_clock_sec", 0.0)
            rtf = r.get("rtf", wall_sec / max(audio_dur, 0.001))
            print(
                f"{s:6d}  {audio_dur:9.1f}  {r['n_words']:6d}  {r['n_repeats']:8d}  "
                f"{wall_sec:8.2f}  {rtf:8.4f}x  ok"
            )
        else:
            err_msg = str(errors.get(s, "unknown error"))
            print(f"{s:6d}  {'':>9}  {'':>6}  {'':>8}  {'':>8}  {'':>9}  ERROR: {err_msg[:30]}")

    print("=" * 80)
    n_ok = summary["succeeded_count"]
    n_err = summary["failed_count"]
    total_audio_hrs = summary["total_audio_sec"] / 3600.0
    total_wall_min = summary["total_wall_sec"] / 60.0
    rtf = summary["overall_rtf"]
    speedup = 1.0 / max(rtf, 1e-6)

    print(
        f"Summary: {n_ok}/{len(surah_list)} surahs succeeded"
        + (f", {n_err} failed" if n_err else "")
    )
    print(
        f"Audio: {total_audio_hrs:.2f}h ({summary['total_audio_sec']:.1f}s) | "
        f"Wall-clock: {total_wall_min:.2f}m ({summary['total_wall_sec']:.1f}s) | "
        f"Overall RTF: {rtf:.4f}x ({speedup:.1f}x realtime speed)"
    )
    print(f"Words aligned: {summary['total_words']:,} | Repeats detected: {summary['total_repeats']:,}")
    print("=" * 80)


def main():
    from .model_manager import resolve_device, resolve_model, resolve_tokens
    args = build_parser().parse_args()
    surah_list = parse_surah_list(args.surahs)
    _validate_device_flags(args)

    device = resolve_device(args.device)
    model_path = resolve_model(args.model, device=device)
    tokens_path = resolve_tokens(args.tokens)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.opus_dir:
        os.makedirs(args.opus_dir, exist_ok=True)

    summary = run_pipelined_batch(
        surah_list,
        args.audio_dir,
        args.out_dir,
        model_path=model_path,
        tokens_path=tokens_path,
        device=device,
        cuda_batch_size=args.cuda_batch_size,
        intra_surah_split=args.intra_surah_split,
        opus_dir=args.opus_dir,
        transcode_opus=args.transcode_opus or (args.opus_dir is not None),
        include_istiaatha=args.include_istiaatha,
        prefetch_workers=args.prefetch_workers,
        prefetch_batches=args.prefetch_batches,
        anomaly_low_ratio=args.anomaly_low_ratio,
        anomaly_high_ratio=args.anomaly_high_ratio,
        ayah_final_high_ratio_mult=args.ayah_final_high_ratio_mult,
        repeat_confidence_margin=args.repeat_confidence_margin,
        max_repeat_window_words=args.max_repeat_window_words,
        tail_silence_sec=args.tail_silence_sec,
        verbose=not args.quiet,
    )

    print_summary_report(summary, surah_list)

    if summary["failed_count"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
