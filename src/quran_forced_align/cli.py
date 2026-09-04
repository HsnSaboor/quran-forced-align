"""S-Tier CLI for Quran Forced Alignment.

Zero-configuration forced-alignment with auto-model resolution, auto-device
detection (CUDA / CPU), Hugging Face caching, and multi-format subtitle output.

Examples:
  # Auto-detects device, auto-downloads model, outputs both JSON and SRT:
  quran-forced-align --surah 66 --audio audio/066_basit.mp3 --out srt_output/066_basit.json

  # Explicit CUDA execution with intra-surah parallelism:
  quran-forced-align --surah 18 --audio audio/018_sudais.mp3 --out out/018 --device cuda --format both
"""
import argparse
import os
import sys
from pathlib import Path

from .constants import (
    DEFAULT_ANOMALY_HIGH_RATIO,
    DEFAULT_ANOMALY_LOW_RATIO,
    DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
    DEFAULT_MAX_REPEAT_WINDOW_WORDS,
    DEFAULT_REPEAT_CONFIDENCE_MARGIN,
    DEFAULT_TAIL_SILENCE_SEC,
)
from .model_manager import resolve_device, resolve_model, resolve_tokens
from .pipeline import align_surah
from .srt import emit_json_rich, emit_srt


def parse_istiaatha_choice(val: str | bool) -> str | bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("auto", "default", "none"):
        return "auto"
    if s in ("yes", "true", "1", "y"):
        return True
    if s in ("no", "false", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid --include-istiaatha choice: '{val}'. Use auto, yes, or no.")


def parse_bismillah_choice(val: str | bool) -> str | bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("auto", "default", "none"):
        return "auto"
    if s in ("yes", "true", "1", "y"):
        return True
    if s in ("no", "false", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid --include-bismillah choice: '{val}'. Use auto, yes, or no.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="quran-forced-align",
        description="Quran Forced Alignment: Align continuous recitation audio with Uthmani Quranic text.",
    )
    ap.add_argument("--surah", type=int, required=True, help="Surah number (1-114)")
    ap.add_argument("--audio", required=True, help="Path to recitation audio (.mp3, .opus, .wav, .m4a)")
    ap.add_argument("--out", required=True, help="Output filepath or base path (e.g. output.json or output.srt)")
    ap.add_argument("--format", choices=["auto", "json", "srt", "both"], default="auto",
                    help="Output format. 'auto' infers from --out extension or writes both if no extension provided.")
    ap.add_argument("--model", default=None,
                    help="Path to ONNX model. If omitted, automatically resolves or downloads from Hugging Face.")
    ap.add_argument("--tokens", default=None,
                    help="Path to tokens.txt. If omitted, automatically resolves or downloads from Hugging Face.")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Execution device: 'auto' (default: CUDA if available, else CPU), 'cuda', or 'cpu'.")
    ap.add_argument("--trt", action="store_true", default=False,
                    help="Enable TensorRT execution provider for FP16 acceleration on NVIDIA GPUs.")
    ap.add_argument("--intra-surah-split", action="store_true", default=None,
                    help="Enable parallel intra-surah GPU streaming. Enabled automatically on CUDA by default.")
    ap.add_argument("--include-istiaatha", type=parse_istiaatha_choice, default="auto",
                    help="Include Isti'adha preamble in reference ('auto', 'yes', or 'no'). Default: 'auto'.")
    ap.add_argument("--include-bismillah", type=parse_bismillah_choice, default="auto",
                    help="Include Bismillah preamble in reference ('auto', 'yes', or 'no'). Default: 'auto'.")
    add_tuning_args(ap)
    return ap


def add_tuning_args(ap: argparse.ArgumentParser) -> None:
    """Shared tuning flags for the forced-alignment + repeat-detection pipeline."""
    ap.add_argument("--anomaly-low-ratio", type=float, default=DEFAULT_ANOMALY_LOW_RATIO,
                    help="Flag a word as anomalously SHORT if duration < ratio * median.")
    ap.add_argument("--anomaly-high-ratio", type=float, default=DEFAULT_ANOMALY_HIGH_RATIO,
                    help="Flag a word as anomalously LONG if duration > ratio * median.")
    ap.add_argument("--ayah-final-high-ratio-mult", type=float, default=DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
                    help="Waqf pause multiplier for ayah-final words before flagging as repeat.")
    ap.add_argument("--repeat-confidence-margin", type=float, default=DEFAULT_REPEAT_CONFIDENCE_MARGIN,
                    help="Acoustic log-likelihood confidence margin for repeat acceptance.")
    ap.add_argument("--max-repeat-window-words", type=int, default=DEFAULT_MAX_REPEAT_WINDOW_WORDS,
                    help="Cap on repeated-phrase search window in words.")
    ap.add_argument("--tail-silence-sec", type=float, default=DEFAULT_TAIL_SILENCE_SEC,
                    help="Padding tail silence in seconds.")


def main():
    args = build_parser().parse_args()

    # 1. Resolve Device & Acceleration Flags
    device = resolve_device(args.device)
    intra_surah_split = (device == "cuda") if args.intra_surah_split is None else args.intra_surah_split

    # 2. Resolve Model & Token Assets (with auto-download if missing)
    tokens_path = resolve_tokens(args.tokens)
    model_path = resolve_model(args.model, device=device)

    print(f"🚀 Quran Forced Alignment: Surah {args.surah}")
    print(f"   Audio:  {args.audio}")
    print(f"   Device: {device.upper()} (Intra-Surah Parallelism: {intra_surah_split})")
    print(f"   Model:  {model_path}")

    # 3. Execute Alignment Pipeline
    records = align_surah(
        args.surah, args.audio,
        model_path=model_path,
        tokens_path=tokens_path,
        device=device,
        intra_surah_split=intra_surah_split,
        include_istiaatha=args.include_istiaatha,
        include_bismillah=args.include_bismillah,
        anomaly_low_ratio=args.anomaly_low_ratio,
        anomaly_high_ratio=args.anomaly_high_ratio,
        ayah_final_high_ratio_mult=args.ayah_final_high_ratio_mult,
        repeat_confidence_margin=args.repeat_confidence_margin,
        max_repeat_window_words=args.max_repeat_window_words,
        tail_silence_sec=args.tail_silence_sec,
    )

    # 4. Route Output Files Cleanly
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    base_stem = out_path.parent / out_path.stem

    fmt = args.format
    if fmt == "auto":
        if ext == ".json":
            fmt = "json"
        elif ext == ".srt":
            fmt = "srt"
        else:
            fmt = "both"

    repeats = sum(1 for r in records if r.get("is_repeat", False))
    print(f"✅ Alignment Complete: {len(records):,} words aligned ({repeats} repeats detected).")

    if fmt in ("json", "both"):
        json_file = str(out_path if ext == ".json" else base_stem.with_suffix(".json"))
        emit_json_rich(records, json_file)
        print(f"   JSON: {json_file} ({os.path.getsize(json_file):,} bytes)")

    if fmt in ("srt", "both"):
        srt_file = str(out_path if ext == ".srt" else base_stem.with_suffix(".srt"))
        emit_srt(records, srt_file)
        print(f"   SRT:  {srt_file} ({os.path.getsize(srt_file):,} bytes)")


if __name__ == "__main__":
    main()
