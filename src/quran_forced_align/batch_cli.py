"""Batch CLI: run the forced-alignment pipeline for multiple surahs in
parallel, one OS process per surah.

  quran-forced-align-batch --surahs 67-71 --audio-dir audio --out-dir srt_output

Each surah's onnxruntime session is deliberately pinned single-threaded/
sequential for determinism (see onnx_model.make_onnx_session). Real OS
threads would fight the GIL on the pure-Python Viterbi loop in viterbi.py
and gain nothing, so this uses ProcessPoolExecutor (genuine CPU-core
parallelism, no shared mutable state) rather than ThreadPoolExecutor.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from .cli import add_tuning_args
from .pipeline import align_surah, align_surahs_batched
from .srt import emit_json_rich, emit_srt


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
    audio_path = os.path.join(audio_dir, f"{surah:03d}.mp3")
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
                            anomaly_low_ratio: float, anomaly_high_ratio: float,
                            ayah_final_high_ratio_mult: float, repeat_confidence_margin: float,
                            max_repeat_window_words: int | None, tail_silence_sec: float) -> list[dict]:
    """Module-level, picklable worker: aligns a BATCH of surahs together
    through `pipeline.align_surahs_batched`'s single batched CUDA
    inference call (see that function's docstring), and writes each
    surah's own SRT + JSON output files. One `ProcessPoolExecutor` task
    per BATCH (not per surah) -- `--cuda-batch-size` controls how many
    surahs each batch/task covers; `--max-workers` still controls how many
    of these batch-tasks run as concurrent OS processes (each opening its
    own CUDA context on the same GPU, same caveat as the unbatched
    `--device cuda` path -- see `build_parser`'s `--max-workers` help).
    """
    t0 = time.monotonic()
    audio_paths = [os.path.join(audio_dir, f"{surah:03d}.mp3") for surah in surahs]

    records_list = align_surahs_batched(
        surahs, audio_paths,
        model_path=model_path,
        tokens_path=tokens_path,
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
        # Per-surah wall_clock_sec is not separately measurable inside a
        # batched call (all surahs in the batch share one inference pass)
        # -- report the WHOLE BATCH's elapsed time on every surah's row
        # instead of a misleading per-surah number, and let the summary
        # table's grouping make clear these surahs shared one batch.
        results.append({
            "surah": surah,
            "n_words": len(records),
            "n_repeats": n_repeats,
            "wall_clock_sec": elapsed,
        })
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surahs", required=True,
                     help='surah numbers to process: a range ("67-71") or a comma list ("67,68,69")')
    ap.add_argument("--audio-dir", required=True,
                     help='directory containing "{surah:03d}.mp3" files (zero-padded 3 digits)')
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="model/zipformer_p_arabic_v2.int8.onnx")
    ap.add_argument("--tokens", default="model/tokens.txt")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                     help="forced-alignment execution engine for every worker process (see "
                          "cli.py's --device for the full description). With --device cuda, "
                          "every worker process opens its own CUDA context on the same GPU, so "
                          "--max-workers should be sized to the GPU's VRAM budget, not "
                          "os.cpu_count() (this flag's default) -- see README for guidance.")
    ap.add_argument("--cuda-batch-size", type=int, default=1,
                     help="--device cuda ONLY: run this many surahs together through ONE "
                          "batched acoustic-model inference pass (pipeline.align_surahs_batched) "
                          "instead of one inference pass per surah, for real GPU throughput on "
                          "the model's own dynamic batch axis. Default 1 (no batching, "
                          "equivalent to running --device cuda without this flag at all). "
                          "Ignored (must be 1) with --device cpu. See README's GPU execution "
                          "section for the measured determinism characterization of batched "
                          "inference before increasing this for a production run.")
    ap.add_argument("--intra-surah-split", action="store_true",
                     help="--device cuda ONLY, and only when --cuda-batch-size is 1 (the two "
                          "features are complementary, not combined in this release): split "
                          "EACH surah's own acoustic-model inference into warm-up-overlapped "
                          "segments at real silence points for real single-surah GPU speedup "
                          "(see cli.py's --intra-surah-split for the full description).")
    ap.add_argument("--max-workers", type=int, default=os.cpu_count())
    add_tuning_args(ap)
    return ap


def _chunked(items, size):
    """Split `items` into consecutive sub-lists of at most `size` each --
    the batching granularity `--cuda-batch-size` controls (the LAST chunk
    may be smaller than `size`, e.g. 10 surahs at batch size 4 gives
    chunks of 4, 4, 2)."""
    return [items[i:i + size] for i in range(0, len(items), size)]


def _validate_device_flags(args) -> None:
    """Cross-flag validation for --device/--cuda-batch-size/--intra-surah-split
    that argparse's own per-flag `choices=`/`type=` can't express (these
    constraints are relationships BETWEEN flags, not properties of a
    single flag's value) -- factored out into its own function (rather
    than inlined in `main()`) so it's unit-testable without invoking the
    real `ProcessPoolExecutor`/engine-construction machinery `main()` goes
    on to run after validation passes (see tests/test_batch_cli.py).
    """
    if args.cuda_batch_size != 1 and args.device != "cuda":
        raise SystemExit("--cuda-batch-size > 1 requires --device cuda")
    if args.cuda_batch_size < 1:
        raise SystemExit("--cuda-batch-size must be >= 1")
    if args.intra_surah_split and args.device != "cuda":
        raise SystemExit("--intra-surah-split requires --device cuda")
    if args.intra_surah_split and args.cuda_batch_size != 1:
        raise SystemExit("--intra-surah-split and --cuda-batch-size > 1 cannot be combined in this release")


def main():
    args = build_parser().parse_args()
    surah_list = parse_surah_list(args.surahs)
    max_workers = min(len(surah_list), args.max_workers)

    _validate_device_flags(args)

    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    errors = {}
    if args.cuda_batch_size > 1:
        # Batched CUDA path: each ProcessPoolExecutor task covers a BATCH
        # of surahs (see _align_batch_of_surahs), not one surah each.
        batches = _chunked(surah_list, args.cuda_batch_size)
        max_workers = min(len(batches), args.max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_batch = {
                pool.submit(
                    _align_batch_of_surahs, batch, args.audio_dir, args.out_dir, args.model, args.tokens,
                    args.anomaly_low_ratio, args.anomaly_high_ratio, args.ayah_final_high_ratio_mult,
                    args.repeat_confidence_margin, args.max_repeat_window_words, args.tail_silence_sec,
                ): batch
                for batch in batches
            }
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                exc = future.exception()
                if exc is not None:
                    for surah in batch:
                        errors[surah] = exc
                    print(f"[batch {batch}] FAILED: {type(exc).__name__}: {exc}")
                else:
                    batch_results = future.result()
                    for result in batch_results:
                        results[result["surah"]] = result
                    print(f"[batch {batch}] {sum(r['n_words'] for r in batch_results)} words total, "
                          f"{batch_results[0]['wall_clock_sec']:.1f}s for the whole batch")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_surah = {
                pool.submit(
                    _align_one_surah, surah, args.audio_dir, args.out_dir, args.model, args.tokens,
                    args.device, args.intra_surah_split,
                    args.anomaly_low_ratio, args.anomaly_high_ratio, args.ayah_final_high_ratio_mult,
                    args.repeat_confidence_margin, args.max_repeat_window_words, args.tail_silence_sec,
                ): surah
                for surah in surah_list
            }
            for future in as_completed(future_to_surah):
                surah = future_to_surah[future]
                exc = future.exception()
                if exc is not None:
                    errors[surah] = exc
                    print(f"[surah {surah:03d}] FAILED: {type(exc).__name__}: {exc}")
                else:
                    result = future.result()
                    results[surah] = result
                    print(f"[surah {surah:03d}] {result['n_words']} words, "
                          f"{result['n_repeats']} repeats, {result['wall_clock_sec']:.1f}s")

    print()
    print("=" * 72)
    print(f"{'surah':>6}  {'words':>6}  {'repeats':>8}  {'seconds':>8}  status")
    for surah in surah_list:
        if surah in results:
            r = results[surah]
            print(f"{surah:6d}  {r['n_words']:6d}  {r['n_repeats']:8d}  {r['wall_clock_sec']:8.1f}  ok")
        else:
            print(f"{surah:6d}  {'':>6}  {'':>8}  {'':>8}  ERROR: {errors[surah]}")
    n_ok = len(results)
    n_err = len(errors)
    print(f"{n_ok}/{len(surah_list)} surahs succeeded"
          + (f", {n_err} failed" if n_err else ""))

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
