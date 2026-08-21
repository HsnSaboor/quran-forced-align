---
license: other
license_name: npl-1.1
license_link: LICENSE
language:
- ar
pipeline_tag: automatic-speech-recognition
extra_gated_prompt: >-
  This model is released under the Quran-Lab No-Profit License 1.1 (NPL-1.1).
  By requesting access you agree: (1) to use it only for purposes that do not
  generate profit, as defined in the LICENSE, (2) not to present its output as
  an authoritative religious ruling on anyone's recitation, and (3) to state
  clearly, in any application built on it, that automatic tajweed feedback can
  be wrong and does not replace a qualified teacher.
extra_gated_fields:
  Name: text
  Affiliation / project: text
  Intended use: text
  I agree to the no-profit license terms (NPL) and the conditions above: checkbox
extra_gated_button_content: Request access
tags:
- quran
- tajweed
- phoneme-recognition
- streaming
- ctc
- zipformer
model-index:
- name: zipformer_p-arabic-v3
  results:
  - task:
      type: automatic-speech-recognition
    dataset:
      name: Quran-Lab/quranic-asr-benchmark (v1.1)
      type: Quran-Lab/quranic-asr-benchmark
    metrics:
    - type: per
      value: 1.43
      name: Phoneme Error Rate, held-out studio reciters
    - type: per
      value: 3.65
      name: Phoneme Error Rate, real phone audio
    - type: per
      value: 9.10
      name: Phoneme Error Rate, unseen reciter
---

# zipformer_p-arabic-v3

A 65.5M-parameter **streaming phoneme recogniser for Qur'an recitation** (Hafs 'an 'Asim), built on
Zipformer2 with a CTC head. It transcribes recitation directly into a 251-symbol Qur'anic phonetic
alphabet that encodes tajweed-relevant distinctions: madd length, gemination, ghunna, ikhfaa,
qalqalah, and the emphatic consonants.

**Design principle: no language model, anywhere.** No transducer, no LM rescoring, no text prior.
An internal or external LM silently corrects a reciter's mistakes, which is precisely what a
recitation-assessment system must never do. Accuracy comes from data and the encoder alone.

## Results

Phoneme Error Rate on the open [Quranic ASR Benchmark](https://huggingface.co/datasets/Quran-Lab/quranic-asr-benchmark)
(v1.1 references), against the previous public release of this line:

| held-out set | Quran Model | zipformer_p-arabic-v2 |
|---|---|---|
| studio reciters, 3 benchmark reciters\* | **1.43%** | 5.19% |
| real phone recordings | **3.65%** | 7.92% |
| unseen professional reciter | **9.10%** | 11.91% |

Measured on the shipped checkpoint (SHA-verified) with the public per-eval manifest; an earlier
revision of this card carried 4.92% for the unseen reciter, which does not reproduce on the shipped
weights and has been corrected.

\* The three studio benchmark reciters were intended to be fully absent from training; a post-release
audit found 24 stray clips of them (0.001% of the 1.85M-row finetune manifest, none of the 600
benchmark clips) introduced by the concatenation builder. Reported as-is for transparency.

The emphatic-consonant gap (ص ض ط ظ error rate minus their plain counterparts), a known weakness of
earlier models, is statistically indistinguishable from zero on all three sets.

On [obadx/qdat_bench](https://huggingface.co/datasets/obadx/qdat_bench) (159 everyday reciters, one
ayah, human-annotated tajweed attributes), madd-length measurement by attribute (RMSE in harakat,
lower is better):

| attribute | Quran Model | muaalem-model-v3_2 |
|---|---|---|
| qalo_alif | **0.210** | 0.449 |
| qalo_waw | 0.514 | 0.456 |
| allam_alif | **0.363** | n/a |
| separate madd (munfasil) | 0.887 | 0.687 |
| madd aared | 1.458 | n/a |

## Honest limitations

These are measured, not hypothetical. Read them before building on the model.

1. **It is trained almost entirely on correct recitation with canonical labels.** When a reciter
   deviates from the rule, the model tends to transcribe the rule rather than the deviation. On
   qdat_bench it reported full ikhfaa for 81 of 94 clips where human annotators heard a plain noon.
   Do not use it, as shipped, as a standalone judge of whether ikhfaa or similar quality-contrasts
   were performed.
2. **Free-choice madd lengths (munfasil, aared) are unreliable in the token stream** because the
   training labels fix them at 4 harakat while reciters legitimately choose 2, 4 or 6. Duration
   grading should be done from forced-alignment timing, not from the emitted symbol run-length.
3. **Children recite into its weakest region**: under-12 reciters score 2 to 3 times worse than
   adults on qdat_bench.
4. **Chunk size matters and depends on domain.** On in-domain recitation, streaming chunks (16 or
   24 frames) match or beat full context. On out-of-domain audio, full context is about 2 PER points
   better. The model was trained only at chunk sizes 8, 16 and 24.

## v3.1 (madd fine-tune)

`zipformer_p_arabic_v3.1.*` is a fine-tune of v3 that retunes madd (elongation) length behaviour,
targeting limitation 2 above. It is the same architecture, the same 99-in/99-out streaming interface
and the same `tokens.txt`, so it is a drop-in replacement at the call site in every format, including
the CoreML packages (whose state-packing manifests are byte-identical to v3's). Every weight differs
from v3, so evaluate it on your own material before switching; duration grading from forced-alignment
timing remains more reliable than run-length counting either way.

## Files

| file | purpose |
|---|---|
| `zipformer_p_arabic_v3.pt` | PyTorch weights (average of the last 3 training epochs) |
| `zipformer_p_arabic_v3.onnx` / `zipformer_p_arabic_v3.int8.onnx` | cache-aware **streaming** CTC export, sherpa-onnx compatible |
| `zipformer_p_arabic_v3.float8.mlpackage` | CoreML, 8-bit weights + fp16 compute, Apple Neural Engine, iOS 18+ (multifunction: `front` and `back`). Recommended for iOS. |
| `zipformer_p_arabic_v3.float16.mlpackage` | CoreML fp16, same graph and interface as float8 |
| `zipformer_p_arabic_v3.1.pt` | **v3.1**, the madd fine-tune (see below): PyTorch weights |
| `zipformer_p_arabic_v3.1.onnx` / `zipformer_p_arabic_v3.1.int8.onnx` | v3.1 streaming CTC export, same interface as v3 |
| `zipformer_p_arabic_v3.1.float8.mlpackage` / `zipformer_p_arabic_v3.1.float16.mlpackage` | v3.1 CoreML ANE packages |
| `packing_front.json`, `packing_back.json` | state blob layout for the CoreML packages, **shared by v3 and v3.1** (identical architecture) |
| `tokens.txt` | symbol table. **Use it.** The raw `phoneme_units.json` ids are offset by one against the CTC output layer (blank is 250, not 0); decoding without this table produces garbage |
| `phoneme_units.json` | tokenizer unit inventory |
| `ordered_quran_phonemes.json` | canonical phonemisation of all 6,236 ayat (for retrieval / grading) |
| `quran_text2phoneme.json` | text-to-phoneme lookup used in evaluation |
| `quran_per_eval.py`, `quran_wer_retrieval.py` | evaluation scripts |
| `decode_with_confidence.py` | CTC decoding with per-symbol confidence |
| `export_quran_streaming_onnx.py` | the exact export script that produced the ONNX files |

## Usage (ONNX, streaming)

The ONNX models are standard sherpa-onnx streaming-CTC zipformer2 exports (`model_type=zipformer2`,
chunk 48 input frames). Feed 16 kHz mono, 80-dim kaldi fbank (povey window). The features must be
kaldi fbank, not a mel-spectrogram: a slaney-mel front end roughly doubles the error rate.

```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("zipformer_p_arabic_v3.int8.onnx")
# see quran_per_eval.py for the full streaming loop incl. cache tensors and fbank settings
```

## Usage (CoreML, Apple Neural Engine)

Both v3 and v3.1 ship two multifunction packages each (iOS 18+), identical in graph and interface and
differing only in weight precision: `*.float8.mlpackage` (8-bit per-channel weights, fp16 compute,
63 MB, recommended) and `*.float16.mlpackage` (124 MB). Everything in this section applies to both
models; they are interchangeable at the call site. Each contains the encoder as two
functions, `front` and `back`, sharing one weight blob. They perform the same cache-aware streaming
computation as the ONNX exports.

**Why two functions.** The ANE compiler has a per-program resource budget; this encoder exceeds it as
one program, and the failure mode is silent CPU fallback. Split at the narrowest point of the network
(a single 24x1x384 tensor crosses), both halves compile for the ANE (front 1,138 of 1,288 ops, back
1,405 of 1,572, measured via MLComputePlan). The same approach Apple uses to chunk its Stable
Diffusion UNet.

**Numerical safety, inherited from the v2 field debugging.** Relative to a naive conversion the graph
carries three fixes: a threshold-compare Swoosh overflow guard (fp16 hardware saturates at 65,504 and
never produces inf, which silently corrupts an inf-compare guard), a max-subtracted log-softmax head
(naive log-sum-exp saturates on the ANE), and no fp32 epsilon parameters (they abort ANE
compilation). The mask-building `reverse` was also eliminated by reversing the ramp constant at
export, removing a CPU island.

**Verification.** Streaming decode over 35 recitations (3 reciters), Mac Neural Engine execution
(`cpuAndNeuralEngine`), each package against its own int8 ONNX: v3.1 is token-identical on 35 of 35
clips (554 tokens); v3 on 34 of 35 (519 tokens), and on that single clip the CoreML output matches
the **fp32** ONNX exactly while the int8 export is the outlier — so the CoreML packages are at least
as faithful to the PyTorch weights as the int8 references. Confidence work should use `margin_peak` from `decode_with_confidence.py`
(margins sampled at emission boundaries are not comparable across runtimes).

**Metadata is embedded.** Both packages carry the streaming contract in CoreML metadata, using the
same keys as the ONNX exports (`model_type`, `T`, `decode_chunk_len`, `left_context_len`) plus the
CoreML specifics (`functions`, `call_order`, `cross_tensor`, `counter_input`, `blank_id`,
`feature_type`, `fbank_options`, `recommended_compute_units`), and the two packing manifests inlined
as `packing_front` / `packing_back` — so the side JSON files are convenient, not required. Every
input and output also carries a description, visible in Xcode's model viewer.

```python
import coremltools as ct
md = ct.models.MLModel("zipformer_p_arabic_v3.float8.mlpackage",
                       skip_model_load=True).get_spec().description.metadata
cfg = dict(md.userDefined)          # T=61, decode_chunk_len=48, blank_id=250, ...
```

**Interface, per 0.48 s chunk.** The 97 cache tensors are packed by shape into a few fp16 blob
tensors (`sg_0 ...`); `packing_front.json` / `packing_back.json` document each blob's members, shapes
and concat axis. Carry blobs whole: every `new_sg_k` output feeds the next call's `sg_k` input,
zero-initialised on the first call. The frame counter `processed_lens` (int32, starts at 0) is fed
to **both** functions unchanged, then updated from front's `new_processed_lens` after the pair has
run — back must see the pre-update value.

| Call | Inputs | Outputs |
|---|---|---|
| `front` | `x` (1, 61, 80) fbank chunk, `processed_lens` (1,) int32, blobs `sg_0..sg_12` | `input_tensor_1439_cast_fp16` (24, 1, 384), `new_processed_lens`, `new_sg_0..new_sg_12` |
| `back` | `input_tensor_1439_cast_fp16` from front, `processed_lens` (pre-update), blobs `sg_0..sg_11` | `log_probs` (1, 12, 251), `new_sg_0..new_sg_11` |

```swift
let cfg = MLModelConfiguration()
cfg.computeUnits = .cpuAndNeuralEngine
cfg.functionName = "front"
let front = try MLModel(contentsOf: compiledURL, configuration: cfg)
cfg.functionName = "back"
let back = try MLModel(contentsOf: compiledURL, configuration: cfg)
// per chunk: front.prediction -> back.prediction, carry new_sg_k -> sg_k
```

Integration notes: `log_probs` arrives with padded strides, so honour `MLMultiArray.strides` rather
than indexing contiguously; `MLMultiArray(shape:dataType:)` returns uninitialised memory, so
explicitly zero the state blobs on the first call; the iOS Simulator does not execute these packages
correctly — validate on macOS or a physical device. On the v2 model of this same architecture the
split float8 pair measured 6.8 ms per chunk on the ANE of an iPhone 15 Pro Max (70x realtime), with
the ANE the fastest configuration outright; v3 timings are expected to match and will be updated
when measured on device.

## Training summary

Trained on about 5,400 effective hours per epoch for 10 epochs: professional complete-mushaf
recitations (36 EveryAyah reciters, 1,161 additional reciters segmented from public recitation
archives), real phone recordings, concatenated multi-ayah context windows, same-ayah repetition
clips, and a broad-Arabic tier, with noise, reverberation, wind, speed and tempo augmentation.
Labels are the deterministic Qur'anic phonetic script of
[quran-transcript](https://github.com/obadx/quran-transcript) (Hafs, murattal).

## License

Released under the **Quran-Lab No-Profit License, Version 1.1 (NPL-1.1)**; see [LICENSE](LICENSE).
In short: free to use, study, adapt and redistribute, for any purpose that does not generate profit.

## Related

- Benchmark and scorer: [Quran-Lab/quranic-asr-benchmark](https://huggingface.co/datasets/Quran-Lab/quranic-asr-benchmark)
- Leaderboard: [Quran-Lab/quranic-asr-leaderboard](https://huggingface.co/spaces/Quran-Lab/quranic-asr-leaderboard)
- Previous releases in this line: [zipformer_p-arabic-v2](https://huggingface.co/Muno459/zipformer_p-arabic-v2), [zipformer_p-quran](https://huggingface.co/Muno459/zipformer_p-quran)
- The qdat_bench tajweed benchmark and the muaalem models by [obadx](https://huggingface.co/obadx), whose
  phonetic script this model's labels are built on
