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
uv sync --extra cpu     # default execution engine -- laptops, CI, any CPU-only machine
```

or, on a CUDA-capable GPU machine (e.g. Colab) for the batch/multi-reciter
workload the `cuda` engine targets:

```bash
uv sync --extra cuda
```

`cpu` and `cuda` are mutually exclusive (`onnxruntime`/`onnxruntime-gpu`
provide the same top-level module and cannot coexist in one environment --
see `pyproject.toml`'s comments) -- install exactly one, matching whichever
`--device` you plan to run with.

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

### GPU execution (`--device cuda`)

```bash
uv run quran-forced-align --surah 2 --audio audio/002.mp3 --out srt_output/002.srt --device cuda
```

Runs the acoustic model on onnxruntime's `CUDAExecutionProvider` and CTC
forced alignment via `torchaudio.functional.forced_align`'s compiled CUDA
kernel, instead of the default CPU engine. Requires `uv sync --extra cuda`
and a CUDA-capable GPU. Output is byte-identical to the CPU engine's
`word`/`start`/`end`/`sura`/`aya`/`is_repeat`/`letters` fields (both engines
align against the exact same reference and produce the exact same
monotonic best-path frame boundaries); `avg_logprob`/`min_decision_margin`
differ slightly between engines by design (each engine's margin is derived
from a different, engine-specific quantity -- see
`engines/cuda.py`'s module docstring) but both are internally consistent,
deterministic across repeated runs on the same engine, and follow the same
"more negative/smaller = less confident" convention.

The default `--model` (the bundled int8-quantized ONNX) works correctly on
both engines -- verified empirically: int8-on-CUDA's greedy-decode argmax
matches int8-on-CPU's for every frame on a real streaming inference run,
and int8-on-CUDA is itself deterministic across repeated runs. If you have
the fp32 ONNX export (`zipformer_p_arabic_v2.onnx`, not bundled in this
repo -- see the model's Hugging Face page) it's numerically closer to the
original PyTorch model, since the int8 quantization was calibrated for
CPU inference; pass `--model path/to/zipformer_p_arabic_v2.onnx` with
`--device cuda` if you want that extra fidelity.

### GPU performance optimizations

Every optimization below was verified on a real Colab T4 GPU session
(not just unit tests) to leave forced-alignment output byte-identical to
the unoptimized baseline (word/timing/repeat-flag fields), and to remain
deterministic across repeated runs. Several candidate optimizations were
investigated and REJECTED after live testing found they either gave no
measurable benefit or broke correctness -- see "Rejected optimizations"
below.

**IO Binding (always on, no flag).** The CUDA engine keeps every one of
the streaming model's ~97 cache tensors resident on the GPU across the
whole chunk loop (`onnxruntime.InferenceSession.io_binding()`), instead of
round-tripping each tensor through host memory on every chunk via plain
`session.run()`. For a long surah's ~14,600 chunks that removes ~2.8
million small, fixed-overhead-dominated host<->device transfers.

**`--cuda-batch-size N` (batch_cli.py only): batch multiple surahs
together.** Runs N surahs' acoustic-model inference through ONE streaming
chunk loop, stacked along the model's own dynamic batch (`N`) axis --
mirroring how sherpa-onnx/icefall batch multiple independent streaming
ASR utterances in production. Measured ~1.5x wall-clock speedup batching
just 2 surahs together on a real T4 GPU, byte-identical word/timing/repeat
output to running them one at a time.

```bash
uv run quran-forced-align-batch --surahs 1-114 --audio-dir audio --out-dir srt_output \
  --device cuda --cuda-batch-size 8 --max-workers 2
```

**`--intra-surah-split` (cli.py and batch_cli.py): split ONE surah's own
inference across silence points.** Finds real pause points in the audio
(energy-based silence detection, see `silence.py`) and splits that ONE
surah's acoustic-model inference into multiple segments, each run
independently and batched together via the same batch-axis mechanism as
`--cuda-batch-size` -- giving real single-surah GPU speedup (measured
1.7x-1.9x across 4 different real surahs on a T4 GPU) where
`--cuda-batch-size` alone cannot help (that flag needs MULTIPLE surahs to
batch; this flag parallelizes within just one).

```bash
uv run quran-forced-align --surah 66 --audio audio/066.mp3 --out srt_output/066.srt \
  --device cuda --intra-surah-split
```

*How it stays correct*: naively resetting the model's streaming cache
mid-recording is NOT safe -- verified empirically: doing so with no
warm-up caused 5 wrong phoneme decisions in just the first 50 frames after
the reset, since the model's cache carries real information from the
actual preceding audio forward, and a hard zero-reset is measurably worse
than "no context" (it's *wrong* context the model never sees in normal
use). This feature avoids that by prepending each split segment with a
generous window (100 chunks, ~48s) of REAL preceding audio as a
throwaway "warm-up" before that segment's first frame is trusted --
exploiting the model's own metadata (`left_context_len`: 256/128/64/32/
64/128 frames per encoder stage, a BOUNDED window, not unbounded history)
so a long-enough warm-up lets the model rebuild an operationally-
equivalent cache from scratch. Verified across 4 real surahs and multiple
split counts (K=2 through 5): zero argmax or forced-alignment differences
in every test, with the 100-chunk warm-up giving 3x+ safety margin over
the smallest warm-up window found sufficient (30 chunks). Splitting at a
genuine silence point (rather than an arbitrary chunk boundary) isn't
required for correctness, but needs a smaller warm-up window to reach
that same zero-difference bar -- and avoids ever landing a split point
mid-word/mid-madd-elongation. Falls back to unsplit inference
automatically if the audio has no usable silence gap (e.g. a short surah,
or one recited with no internal pause at all, like Ayat al-Kursi).

**Rejected optimizations** (tested live, not adopted):
- *CUDA Graphs*: confirmed via live testing to produce SILENTLY WRONG
  output for this model's recurrent cache-threading -- capture/replay
  works for the first two calls but every subsequent chunk's log-probs
  diverge (verified: 30 argmax flips by the 6th chunk in one test). CUDA
  Graphs assumes stable, non-aliasing memory access patterns that don't
  hold for this model's many small "swap read/write role every call"
  cache tensors. Not used.
- *Relaxing `intra_op_num_threads`/`inter_op_num_threads` for the CUDA
  EP*: measured no timing difference (noise-level, <2%) since this
  model's per-chunk graph is tiny and launch-overhead-bound, not CPU-
  scheduling-bound. `ORT_PARALLEL` execution mode was additionally found
  to HANG indefinitely for this graph on a live test -- both settings
  remain pinned to the original single-threaded/sequential CPU-EP-derived
  values on the CUDA EP too, not because they matter for determinism
  there (they don't, GPU kernel reduction order doesn't depend on ORT's
  CPU-side thread count) but because relaxing them gave no benefit worth
  the churn.
- *Multi-stream (`torch.cuda.Stream`) overlap for repeat-detection's
  K-search loop*: measured no speedup (5 sequential small `forced_align`
  calls: ~11.7ms; the same 5 calls on separate streams: ~12.2ms, slightly
  SLOWER) -- each call is already one efficient, low-occupancy kernel
  with no idle GPU capacity for concurrent streams to fill at this scale.

### Output format

The JSON output is a list of per-word records, sorted by start time:

```json
{
  "word": "أَعُوذُ", "start": 0.0, "end": 0.15, "sura": 1, "aya": 0,
  "is_repeat": false, "avg_logprob": -14.26, "min_decision_margin": 6.98, "low_confidence": false,
  "letters": [
    {"char": "أ", "deleted": false, "tajweed_rules": [], "boundary_tajweed_rules": [],
     "start": 0.0, "end": 0.0, "phonemes": [{"phoneme": "ءَ", "start": 0.0, "end": 0.0}]},
    "... one entry per Uthmani character of the word ..."
  ]
}
```

- `word`/`start`/`end`/`sura`/`aya`/`is_repeat` are the same flat fields
  this package has always emitted (the web player reads only these, so
  older `srt_output/*.json` files and this format are mutually readable).
- `avg_logprob`/`min_decision_margin`/`low_confidence`: per-word
  alignment-confidence signals, computed as free post-processing over data
  the Viterbi pass already produces -- no extra ONNX inference, no extra
  DP pass, no multi-beam re-alignment. `avg_logprob` is `null` and
  `min_decision_margin` can be `null` for a degenerate/unambiguous span
  (both are `+inf`/`-inf` internally, sanitized to `null` at JSON-write
  time since strict JSON has no infinity literal). See `confidence.py`'s
  module docstring.
- `letters`: one entry per Uthmani character of the word, each with the
  madd/qalqalah tajweed rule(s) `quran_transcript` tags that character
  with (`tajweed_rules`), whether it's silently dropped (`deleted`, e.g.
  hamzat al-wasl), any additional tajweed rule that only applies under
  continuous non-pausing recitation across an ayah boundary
  (`boundary_tajweed_rules` -- populated only at the first/last letter of
  an ayah-boundary word), and its own phoneme-level timing
  (`start`/`end`/`phonemes`). A `deleted` letter or a diacritic-only
  letter (haraka/tanween/shadda -- its timing is inherently its base
  letter's timing, since this model's phoneme tokens are pre-composed
  base+diacritic clusters) gets `start`/`end` = `null` and an empty
  `phonemes` list.
- Ghunnah/idgham/ikhfaa are NOT currently tag-observable this way: the
  underlying `quran_transcript` package applies them to the phoneme TEXT
  (so the model still hears/aligns the correct merged sound) but does not
  attach a `tajweed_rules` tag for them the way it does for madd/
  qalqalah -- only madd and qalqalah rules are populated today.

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

Add `--device cuda` (requires `uv sync --extra cuda`) to run every worker
process's forced alignment on the GPU engine instead. Each worker process
opens its own CUDA context on the same GPU, so size `--max-workers` to the
GPU's VRAM budget for this workload, not `os.cpu_count()` (this flag's
default, tuned for the CPU engine):

```bash
uv run quran-forced-align-batch \
  --surahs 1-114 \
  --audio-dir audio \
  --out-dir srt_output \
  --device cuda \
  --max-workers 2
```

See "GPU performance optimizations" above for `--cuda-batch-size` (batch
multiple surahs together through one inference pass) and
`--intra-surah-split` (split each surah's own inference across silence
points) -- both work with `quran-forced-align-batch`, and CAN be combined
for maximum GPU throughput: every surah in a `--cuda-batch-size` batch is
ALSO split into its own silence-based segments, and every surah's every
segment is flattened into one combined batched inference call:

```bash
uv run quran-forced-align-batch \
  --surahs 1-114 \
  --audio-dir audio \
  --out-dir srt_output \
  --device cuda \
  --cuda-batch-size 8 \
  --intra-surah-split \
  --max-workers 2
```

## Web player

`web-player/index.html` is a standalone (no build step) HTML/JS/Tailwind
page that plays a surah's audio with live word-level highlighting, driven
directly by `srt_output/*.json` + `audio/*` -- see that file's own comments
for the highlighting/repeat-detection/timeline UI logic.

Serve the repo root with **`web-player/serve.py`**, not plain
`python -m http.server`:

```bash
python3 web-player/serve.py 8000   # then open http://localhost:8000/web-player/
```

`python -m http.server` does not support HTTP Range requests (no
`Accept-Ranges` header, always `200` never `206`) -- Chrome's `<audio>`
element reports `seekable.length === 0` for any file served that way, so
clicking a word or the timeline silently fails to actually seek even
though the file loads and plays from the start. `serve.py` is a small
Range-supporting drop-in replacement for local development; any real web
server (nginx, Caddy, GitHub Pages, a CDN, etc.) already supports Range
requests correctly, so this only matters when serving locally.

## Performance

The whole-surah CTC Viterbi forced-alignment DP (`viterbi.py`) picks
between two internally-identical implementations based on problem size:
a direct (T,M) array for anything under `_DIRECT_PATH_MAX_CELLS`, and an
exact checkpointed (Hirschberg-style) backtrace for anything larger, which
only ever holds O(sqrt(T)*M) of the trellis in memory instead of the full
O(T*M). This matters for Al-Baqarah (surah 2, ~6122 words, up to
~215,000 audio frames): the naive full-array approach would need ~69GB of
RAM just for the `alpha` array alone -- infeasible on ordinary hardware.
The checkpointed path handles the same problem in well under 1GB, at the
cost of roughly 2x the forward-pass FLOPs (re-deriving each chunk's rows
once during the initial pass, once more during backtrace reconstruction).
Both paths are verified byte-identical to each other (and to the original
un-optimized reference implementation) across a battery of sizes including
every internal chunk-boundary edge case -- see `viterbi.py`'s own
docstring and the equivalence checks referenced there.

Also optimized (all verified byte-identical to the pre-optimization
baseline on surahs 66-72, confirmed via SHA-256 hash comparison):
eliminating the `backptr` array (recomputed on the fly during backtrace via
deterministic-argmax replay, since arithmetic here is pinned single-threaded
float64), removing per-audio-frame array allocations in the Viterbi forward
step (`_step_alpha`'s in-place scratch buffers), passing the numpy sample
array directly into `kaldi_native_fbank` instead of boxing every sample via
`.tolist()`, preallocating feature/log-prob arrays instead of
list-then-stack/concatenate, and piping ffmpeg's decoded PCM straight from
its stdout instead of a temp-file round-trip.

Net effect on a representative surah (66, 269 words): ~79s -> ~52s
end-to-end (~34% faster). Net effect on Al-Baqarah: went from impossible
(OOM/swap-thrash on a 7.6GB-RAM machine) to completing in a few minutes with
comfortable memory headroom -- see the Al-Baqarah row in the verification
table below.

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

### Large-scale run: Al-Baqarah (surah 2, 3 reciters)

The single largest surah in the Quran (286 ayahs, 6122 words) was run
end-to-end against 3 different full-surah recordings, exercising the
checkpointed-Viterbi path described above at real scale (not just synthetic
benchmarks):

| reciter | audio length | words | repeats flagged | monotonic violations |
|---------|-------------|-------|------------------|-----------------------|
| Ahmed Kaseb | ~117.6 min | 6142 (6122 unique) | 9 sites (20 word-cues) | 0 |
| Abdullah Ali Jabir | ~130.6 min | 6173 (6122 unique) | ~26 sites (51 word-cues) | 0 |
| Mohammad Ayoub | ~143.7 min | 6337 (6122 unique) | ~57 sites (215 word-cues) | 0 |

All 3 completed without exhausting memory on a 7.6GB-RAM machine (the
un-optimized pipeline could not even start the Viterbi pass at this size --
see "Performance" above). Output sanity-checked (monotonic non-overlapping
timestamps, last word's end time matching the source audio's real
duration) and every flagged site across all 3 reciters (9 + ~26 + ~57 = 92
sites total) has since been manually spot-checked by ear -- all confirmed
genuine, 0 false positives found.

### Surah 35 (Fatir), Abdullah Mohsin Al-Kasim (murattal)

| reciter | audio length | words | repeats flagged | monotonic violations |
|---------|-------------|-------|------------------|-----------------------|
| Abdullah Mohsin Al-Kasim | ~12.9 min | 780 | 0 | 0 |

A standard (non-hifz-practice) murattal recitation -- 0 repeats flagged is
expected here, not a detector gap.

### Surah 7 (Al-A'raf), Hammad Sinan (murattal)

| reciter | audio length | words | repeats flagged | monotonic violations |
|---------|-------------|-------|------------------|-----------------------|
| Hammad Sinan | ~66.3 min | 3377 (3325 unique) | 19 sites (52 word-cues) | 0 |

Labeled "murattal" but unlike surah 35's run above, this one DID get
repeats flagged -- all 19 sites have since been manually spot-checked by
ear and confirmed genuine, 0 false positives found. So "murattal" is not
a reliable predictor of whether repeats occur; the detector's flags held
up regardless.

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
uv sync --extra cpu --group dev   # pytest lives in the separate `dev` dependency group
uv run pytest -v
```

Requires the model file to be present at `model/zipformer_p_arabic_v2.int8.onnx`
(tests are skipped with a clear message otherwise).
