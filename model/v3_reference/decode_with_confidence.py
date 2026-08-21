"""Per-phoneme confidence, alternatives and timings for zipformer_p-arabic-v2.

Greedy CTC throws away almost everything the model knows: it keeps the argmax and discards the full
251-way posterior. This script keeps it, so every emitted phoneme carries

  - confidence   : probability of the chosen unit (mean over the frames that emitted it)
  - alternatives : the next most likely units with their probabilities
  - start / end  : time span in seconds
  - margin       : confidence minus the runner-up, i.e. how decisive the choice was

Why this matters for recitation grading: "the model was unsure" and "the reciter said something
different" look identical in a plain transcript. With confidence you can treat low-confidence
positions as neutral and only flag a mistake when the model is confident AND disagrees.

Usage:  python decode_with_confidence.py audio.wav [--topk 3] [--json out.json]
"""
import sys, os, json, argparse, warnings
warnings.filterwarnings("ignore")
import torch, numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
# repo layout: scripts/ next to this file, or point ICEFALL_ROOT at a checkout
ROOT = os.environ.get("ZP_ROOT", HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from zipformer_rnnt_ctc_train import (import_training_deps, build_model,
                                      build_context_profiles, kaldi_fbank_batch)
from pathlib import Path
import_training_deps(Path(ROOT))

FRAME_SEC = 0.04            # 10 ms fbank hop x 4 total subsampling = 40 ms per encoder frame

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load(ckpt_path, units_path):
    ck = torch.load(ckpt_path, map_location=DEVICE)
    sd, blank = ck["model"], ck["blank_id"]
    raw = json.load(open(units_path, encoding="utf-8"))
    id2u = sorted((u for u in raw if u != "<blank>"), key=lambda u: raw[u])
    cf = sorted({p.zipformer_chunk_frames for p in build_context_profiles("1000:0.5,640:0.35,320:0.15")})
    m = build_model(sd["ctc_head.weight"].shape[0], blank, cf, 256).to(DEVICE).eval()
    m.load_state_dict(sd, strict=False)
    m.encoder.chunk_size = (24,); m.encoder.left_context_frames = (256,)
    return m, blank, id2u

def decode(m, blank, id2u, wav, topk=3):
    x = torch.from_numpy(wav)[None, :].to(DEVICE)
    feats, fl = kaldi_fbank_batch(x, torch.tensor([x.shape[1]]).to(DEVICE))
    with torch.inference_mode():
        enc, el = m.encode(feats, fl)
        probs = m.ctc_head(enc).softmax(-1)[0, :int(el)].cpu()      # (T, V) real probabilities
    best = probs.argmax(-1)
    out, prev, run = [], -1, []
    def flush():
        if not run: return
        fr = [f for f, _ in run]; unit = run[0][1]
        p = probs[fr]                                          # (n, V) frames that emitted this unit
        conf = float(p[:, unit].mean())
        avg = p.mean(0)
        order = torch.argsort(avg, descending=True)
        alts = [{"unit": id2u[int(i)] if int(i) != blank else "<blank>",
                 "p": round(float(avg[int(i)]), 4)}
                for i in order[:topk + 1] if int(i) != unit][:topk]
        runner = alts[0]["p"] if alts else 0.0
        # Peak-frame margin: evaluated at the frame where this unit is most
        # probable within its emitted span. Unlike first-frame or span-mean
        # statistics, this is stable across runtimes: greedy emission
        # boundaries can shift by one frame between engines (ONNX, CoreML,
        # PyTorch) on near-tie transition frames, which makes boundary-frame
        # margins collapse spuriously while the underlying posteriors agree.
        # Measured across 33 recitations, peak-frame margins agree between
        # ONNX int8 and CoreML to within 0.11 worst-case (boundary-frame
        # margins diverged up to 0.97 on identical decodes). Threshold
        # decisions (e.g. mistake flagging) should use margin_peak.
        pk = int(torch.argmax(p[:, unit]))
        pk_sorted = torch.sort(p[pk], descending=True).values
        margin_peak = float(pk_sorted[0] - pk_sorted[1])
        out.append({"phoneme": id2u[unit], "confidence": round(conf, 4),
                    "margin": round(conf - runner, 4),
                    "confidence_peak": round(float(p[pk, unit]), 4),
                    "margin_peak": round(margin_peak, 4),
                    "start": round(fr[0] * FRAME_SEC, 3),
                    "end": round((fr[-1] + 1) * FRAME_SEC, 3),
                    "frames": len(fr), "alternatives": alts})
    for f, u in enumerate(best.tolist()):
        if u == prev:
            if u != blank: run.append((f, u))
            continue
        flush(); run = [] if u == blank else [(f, u)]
        prev = u
    flush()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--json", default="")
    ap.add_argument("--checkpoint", default=os.path.join(HERE, "zipformer_p_arabic_v2.pt"))
    ap.add_argument("--units", default=os.path.join(HERE, "phoneme_units.json"))
    ap.add_argument("--low", type=float, default=0.50, help="flag phonemes below this confidence")
    a = ap.parse_args()

    w, sr = sf.read(a.audio, dtype="float32")
    if getattr(w, "ndim", 1) > 1: w = w.mean(1)
    assert sr == 16000, f"expected 16 kHz audio, got {sr}"
    m, blank, id2u = load(a.checkpoint, a.units)
    res = decode(m, blank, id2u, w, a.topk)

    print("".join(r["phoneme"] for r in res), "\n")
    print(f"{'time':>13}  {'ph':<6} {'conf':>6} {'margin':>7} {'m_peak':>7}  alternatives")
    for r in res:
        alts = "  ".join(f"{x['unit']}:{x['p']:.2f}" for x in r["alternatives"])
        flag = "  <-- low" if r["confidence"] < a.low else ""
        print(f"{r['start']:6.2f}-{r['end']:5.2f}  {r['phoneme']:<6} {r['confidence']:6.3f} "
              f"{r['margin']:7.3f} {r['margin_peak']:7.3f}  {alts}{flag}")
    n_low = sum(1 for r in res if r["confidence"] < a.low)
    print(f"\n{len(res)} phonemes, {n_low} below confidence {a.low} "
          f"({100*n_low/max(len(res),1):.1f}%), mean confidence {np.mean([r['confidence'] for r in res]):.3f}")
    if a.json:
        json.dump(res, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("wrote", a.json)

if __name__ == "__main__":
    main()
