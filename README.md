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
duration) but NOT yet manually spot-checked by ear the way surahs 66-72
were -- treat the `is_repeat` flags for surah 2 as unverified until that
pass is done, same caveat as the 0-flag surahs above.

### Surah 35 (Fatir), Abdullah Mohsin Al-Kasim (murattal)

| reciter | audio length | words | repeats flagged | monotonic violations |
|---------|-------------|-------|------------------|-----------------------|
| Abdullah Mohsin Al-Kasim | ~12.9 min | 780 | 0 | 0 |

A standard (non-hifz-practice) murattal recitation -- 0 repeats flagged is
expected here, not a detector gap.

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
