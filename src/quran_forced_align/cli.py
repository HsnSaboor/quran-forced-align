"""Thin argparse wrapper around pipeline.align_surah for single-surah use.

  quran-forced-align --surah 1 --audio audio/001001_full.mp3 --out srt_output/forced_test_s1.srt
"""
import argparse
import os

from .constants import (
    DEFAULT_ANOMALY_HIGH_RATIO,
    DEFAULT_ANOMALY_LOW_RATIO,
    DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
    DEFAULT_MAX_REPEAT_WINDOW_WORDS,
    DEFAULT_REPEAT_CONFIDENCE_MARGIN,
    DEFAULT_TAIL_SILENCE_SEC,
)
from .pipeline import align_surah
from .srt import emit_json_rich, emit_srt


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surah", type=int, required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="model/zipformer_p_arabic_v2.int8.onnx")
    ap.add_argument("--tokens", default="model/tokens.txt")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                     help="forced-alignment execution engine: 'cpu' (default, deterministic ONNX "
                          "CPUExecutionProvider + numpy Viterbi) or 'cuda' (onnxruntime "
                          "CUDAExecutionProvider + torchaudio.functional.forced_align; requires "
                          "the 'cuda' extra -- see pyproject.toml -- and a CUDA-capable GPU)")
    add_tuning_args(ap)
    return ap


def add_tuning_args(ap: argparse.ArgumentParser) -> None:
    """Shared tuning flags for the forced-alignment + repeat-detection
    pipeline, factored out so cli.py and batch_cli.py don't duplicate flag
    definitions."""
    ap.add_argument("--anomaly-low-ratio", type=float, default=DEFAULT_ANOMALY_LOW_RATIO,
                     help="flag a word as anomalously SHORT if its duration is below this fraction of the median")
    ap.add_argument("--anomaly-high-ratio", type=float, default=DEFAULT_ANOMALY_HIGH_RATIO,
                     help="flag a word as anomalously LONG if its duration exceeds this multiple of the median")
    ap.add_argument("--ayah-final-high-ratio-mult", type=float, default=DEFAULT_AYAH_FINAL_HIGH_RATIO_MULT,
                     help="multiply --anomaly-high-ratio by this for ayah-final words before flagging them as "
                          "anomalously long, since natural waqf (pause) lengthening at ayah boundaries is "
                          "expected and NOT a repeat signal (see repeats.detect_and_fix_repeats)")
    ap.add_argument("--repeat-confidence-margin", type=float, default=DEFAULT_REPEAT_CONFIDENCE_MARGIN,
                     help="reject a candidate repeat split unless BOTH copies' average per-frame log-likelihood "
                          "along the doubled-reference re-alignment is within this margin of the surah's own "
                          "normal-word acoustic-confidence baseline (see repeats.detect_and_fix_repeats)")
    ap.add_argument("--max-repeat-window-words", type=int, default=DEFAULT_MAX_REPEAT_WINDOW_WORDS,
                     help="OPTIONAL hard cap (in words) on the repeated-phrase K-search when a word's duration is "
                          "flagged as anomalous. By default (unset) the search is bounded naturally by how many "
                          "words remain in the current ayah -- a real hifz-practice repeat never spans into a "
                          "different ayah -- so any repeated-phrase length is found correctly with no magic "
                          "number required; this flag only exists as a cost-control escape valve for "
                          "pathologically long ayahs (see repeats.detect_and_fix_repeats)")
    ap.add_argument("--tail-silence-sec", type=float, default=DEFAULT_TAIL_SILENCE_SEC)


def main():
    args = build_parser().parse_args()

    records = align_surah(
        args.surah, args.audio,
        model_path=args.model,
        tokens_path=args.tokens,
        device=args.device,
        anomaly_low_ratio=args.anomaly_low_ratio,
        anomaly_high_ratio=args.anomaly_high_ratio,
        ayah_final_high_ratio_mult=args.ayah_final_high_ratio_mult,
        repeat_confidence_margin=args.repeat_confidence_margin,
        max_repeat_window_words=args.max_repeat_window_words,
        tail_silence_sec=args.tail_silence_sec,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    emit_srt(records, args.out)
    json_out = os.path.splitext(args.out)[0] + ".json"
    emit_json_rich(records, json_out)


if __name__ == "__main__":
    main()
