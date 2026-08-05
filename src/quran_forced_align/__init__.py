"""quran-forced-align: CTC forced alignment for Quran surah audio.

Given a surah number + an audio file of its recitation, produces word-level
timed SRT + JSON output. Uses a raw-ONNX Zipformer2-CTC streaming model with
manual cache-threading (bypassing sherpa_onnx's limited Python API, which
does not expose the full per-frame log-probability matrix forced alignment
needs) and a whole-surah-at-once Viterbi forced-alignment pass, plus a
two-stage repeat-detection system (anomaly-duration gate + K-window search +
acoustic-confidence gate + gap-artifact reject) for hifz-practice word
repeats.

See `quran_forced_align.pipeline` for the full architectural rationale
(why forced alignment instead of free-decode-then-match, why raw ONNX
instead of sherpa_onnx's Python API, why whole-surah-at-once Viterbi, and
the determinism guarantees this package preserves).
"""
