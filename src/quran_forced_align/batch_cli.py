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
from .pipeline import align_surah
from .srt import emit_json, emit_srt


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

    cue_tuples = align_surah(
        surah, audio_path,
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
    emit_srt(cue_tuples, out_path)
    json_out = os.path.splitext(out_path)[0] + ".json"
    emit_json(cue_tuples, json_out)

    n_repeats = sum(1 for c in cue_tuples if c[5])
    return {
        "surah": surah,
        "n_words": len(cue_tuples),
        "n_repeats": n_repeats,
        "wall_clock_sec": time.monotonic() - t0,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surahs", required=True,
                     help='surah numbers to process: a range ("67-71") or a comma list ("67,68,69")')
    ap.add_argument("--audio-dir", required=True,
                     help='directory containing "{surah:03d}.mp3" files (zero-padded 3 digits)')
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="model/zipformer_p_arabic_v2.int8.onnx")
    ap.add_argument("--tokens", default="model/tokens.txt")
    ap.add_argument("--max-workers", type=int, default=os.cpu_count())
    add_tuning_args(ap)
    return ap


def main():
    args = build_parser().parse_args()
    surah_list = parse_surah_list(args.surahs)
    max_workers = min(len(surah_list), args.max_workers)

    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    errors = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_surah = {
            pool.submit(
                _align_one_surah, surah, args.audio_dir, args.out_dir, args.model, args.tokens,
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
