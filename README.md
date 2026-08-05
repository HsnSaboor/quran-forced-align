# quran-forced-align

CTC forced alignment for Quran surah audio: given a surah number and an
audio file of its recitation, produces word-level timed SRT + JSON output.
Uses a raw-ONNX Zipformer2-CTC streaming model with manual cache-threading
(bypassing sherpa_onnx's Python API, which does not expose the full
per-frame CTC log-probability matrix that forced alignment needs) and a
whole-surah-at-once Viterbi forced-alignment pass against the known
reference phoneme sequence, plus a repeat-detection system (anomaly-duration
gate + K-window search + acoustic-confidence gate + gap-artifact reject)
that recovers word/phrase repeats from hifz (memorization) practice
recitations. Every source of run-to-run nondeterminism (thread scheduling,
fbank dithering, DP reduction order) is pinned, so alignment output is
byte-identical across repeated runs on the same input. See
`src/quran_forced_align/pipeline.py`'s module docstring for the full
architectural rationale.

## Install

```bash
uv sync
```

The bundled acoustic model (`model/zipformer_p_arabic_v2.int8.onnx`, ~73MB,
gitignored) must be present on disk for any alignment to run -- see "Model
license" below.

## Usage: single surah

```bash
uv run quran-forced-align \
  --surah 1 \
  --audio audio/001001_full.mp3 \
  --out srt_output/001.srt
```

Writes `srt_output/001.srt` and `srt_output/001.json`. Tuning flags
(`--anomaly-low-ratio`, `--anomaly-high-ratio`,
`--ayah-final-high-ratio-mult`, `--repeat-confidence-margin`,
`--max-repeat-window-words`, `--tail-silence-sec`) control the
repeat-detection sensitivity -- see `quran-forced-align --help`.

## Usage: batch (multiple surahs, parallel processes)

```bash
uv run quran-forced-align-batch \
  --surahs 67-71 \
  --audio-dir audio \
  --out-dir srt_output \
  --max-workers 4
```

`--surahs` accepts either a range (`67-71`) or a comma list (`67,68,69`).
`--audio-dir` must contain `{surah:03d}.mp3` files (zero-padded 3 digits,
e.g. `067.mp3`). Each surah is aligned in its own OS process
(`ProcessPoolExecutor`, not threads -- each surah's onnxruntime session is
pinned single-threaded for determinism, so real CPU-core parallelism comes
from separate processes, not GIL-contending threads). One surah failing
does not abort the batch; a summary table is printed at the end.

## Verification status

Batch-run and manually spot-checked against real Alafasy recitation audio
for surahs 66-72 (7 surahs, 1892 words total). Every `is_repeat`-flagged
cue produced by the pipeline in this range was listened to against the
source audio and confirmed genuine -- 0 false positives found:

| surah | words | repeats flagged | verified by ear |
|-------|-------|------------------|------------------|
| 66    | 269   | 7 sites (15 word-cues) | yes -- all 7 genuine |
| 67    | 341   | 3 (ayah 28)      | yes -- genuine |
| 68    | 305   | 0                | n/a |
| 69    | 263   | 0                | n/a |
| 70    | 222   | 0                | n/a |
| 71    | 235   | 4 (ayah 7)       | yes -- genuine |
| 72    | 294   | 1 site (4 word-cues) | yes -- genuine |

Surah 66's 7 repeat sites span the full K-window range the repeat-detector
searches (single-word repeats like مُسْلِمَـٰتٍۢ/يَقُولُونَ/وَنَجِّنِى at
K=1, up to a 5-word phrase repeat at K=5) -- exactly the case shapes
earlier false-positive bugs used to trip on -- and all were confirmed
genuine, which is the strongest evidence gathered so far that the
gap-artifact reject + free-decode cross-check (see `repeats.py`'s
docstring) generalizes beyond the surahs they were calibrated against.

Recall (missed repeats) is still NOT independently audited beyond the two
synthetic ground-truth fixtures (`test_B`/`test_C`) and the confirmed-zero
surahs above -- a genuine no-pause repeat that never trips the initial
duration-anomaly gate would still be silently missed. Treat 0-flag surahs
as "no repeat detected," not "confirmed no repeat."

## Model license

The bundled acoustic model (`Muno459/zipformer_p-arabic-v2`) carries a
"FREE / NON-COMMERCIAL USE LICENSE (No Monetization)": free to use, modify,
and redistribute only within apps/services that are free to their end
users -- no paid tiers, paywalls, ads, or other revenue derived from an app
that uses the model or its outputs. See the model's Hugging Face repo for
the authoritative license text before using this in any commercial
context.

## Tests

```bash
uv run pytest -v
```

Requires the model file to be present at `model/zipformer_p_arabic_v2.int8.onnx`
(tests are skipped with a clear message otherwise).
