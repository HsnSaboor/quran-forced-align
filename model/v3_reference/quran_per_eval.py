"""PER (phoneme error rate) of the Quran-finetuned model on the held-out quranic-asr-benchmark.
Pipeline: label text -> deterministic quran_phonetizer -> gold phoneme UNITS (via the model's
tokenizer);  audio -> encoder -> CTC greedy -> pred phoneme UNITS.  PER = unit edit-distance / gold units.
"""
import sys, io, os, json, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
ROOT = "C:/Users/Anon/research/tarteel-asr"
sys.path.insert(0, ROOT + "/scripts")
from zipformer_rnnt_ctc_train import import_training_deps, build_model, build_context_profiles, kaldi_fbank_batch, load_tokenizer
from zipformer_rnnt_ctc_eval import greedy_ctc_decode, edit_distance
import torch, soundfile as sf, numpy as np, re, unicodedata

# same normalization the training phoneme table was keyed by (build_quran_phonemes_deterministic.norm)
_DIAC = re.compile(r'[ؐ-ًؚ-ٰٟۖ-ۭـ]')
def norm(s):
    s = unicodedata.normalize("NFC", str(s)); s = _DIAC.sub('', s)
    for a, b in [('أ','ا'),('إ','ا'),('آ','ا'),('ٱ','ا'),('ى','ي'),('ة','ه'),('ؤ','و'),('ئ','ي')]:
        s = s.replace(a, b)
    return re.sub(r'\s+', ' ', s).strip()

def fix(p): return p.replace("/mnt/c/", "C:/").replace("/mnt/C/", "C:/")

def read_wav(p):
    d, sr = sf.read(p, dtype="float32", always_2d=False)
    if d.ndim > 1: d = d.mean(1)
    return torch.from_numpy(np.ascontiguousarray(d)), sr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=ROOT + "/data/benchmark/benchmark_manifest.jsonl")
    ap.add_argument("--checkpoint", default=ROOT + "/checkpoints/zipformer-phoneme-quran-finetune/quran_finetune_final.pt")
    ap.add_argument("--tokenizer", default=ROOT + "/data/zipformer_rnnt_ctc/tokenizer/phoneme_units.json")
    ap.add_argument("--context-mix", default="1000:0.5,640:0.35,320:0.15")
    ap.add_argument("--left-context-frames", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=0, help="force encoder chunk_size in frames (0=max of context-mix=24=1000ms lookahead)")
    ap.add_argument("--enc-batch", type=int, default=8)
    ap.add_argument("--collapse", action="store_true", help="collapse consecutive identical units in pred+gold (removes madd-length diffs -> phoneme-identity PER)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-duration", type=float, default=30.0)
    args = ap.parse_args()
    import_training_deps(Path(ROOT)); dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = load_tokenizer(args.tokenizer)
    blank_id = tok.get_piece_size(); vocab = blank_id + 1
    profiles = build_context_profiles(args.context_mix)
    cf = sorted({p.zipformer_chunk_frames for p in profiles})
    model = build_model(vocab, blank_id, cf, args.left_context_frames).to(dev).eval()
    ck = torch.load(args.checkpoint, map_location=dev); model.load_state_dict(ck["model"])
    _chunk = args.chunk if args.chunk > 0 else max(cf)
    model.encoder.chunk_size = (_chunk,); model.encoder.left_context_frames = (args.left_context_frames,)
    print(json.dumps({"checkpoint_step": ck.get("step"), "device": dev, "chunk": _chunk}), flush=True)

    # deterministic gold phonemes via the training norm-keyed text->phoneme table
    raw = json.load(open(ROOT + "/data/zipformer_rnnt_ctc/quran_text2phoneme.json", encoding="utf-8"))
    table = {norm(k): v for k, v in raw.items()}
    print(f"phoneme table: {len(raw)} raw -> {len(table)} norm keys", flush=True)

    rows = []
    for ln in open(args.manifest, encoding="utf-8"):
        o = json.loads(ln); o["audio_filepath"] = fix(o["audio_filepath"])
        rows.append(o)
    if args.limit: rows = rows[:args.limit]

    gold = {}; miss = 0; noaud = 0
    for i, o in enumerate(rows):
        ph = table.get(norm(o["text"]))
        if ph is None: miss += 1; continue
        if not os.path.exists(o["audio_filepath"]): noaud += 1; continue
        g = tok.encode(ph)
        if g: gold[i] = g
    idx = [i for i in range(len(rows)) if i in gold]
    print(f"clips: {len(rows)} usable: {len(idx)} table-miss: {miss} no-audio: {noaud}", flush=True)

    tot_err = tot_len = seen = 0
    perfect = 0
    from collections import defaultdict
    src_err = defaultdict(int); src_len = defaultdict(int); src_n = defaultdict(int); src_perf = defaultdict(int)
    with torch.inference_mode():
        for b0 in range(0, len(idx), args.enc_batch):
            bi = idx[b0:b0 + args.enc_batch]
            wavs = []
            for i in bi:
                w, sr = read_wav(rows[i]["audio_filepath"]); wavs.append(w)
            lens = torch.tensor([w.numel() for w in wavs], dtype=torch.long)
            T = int(lens.max()); wp = torch.zeros(len(wavs), T, dtype=torch.float32)
            for k, w in enumerate(wavs): wp[k, :w.numel()] = w
            wp, lens = wp.to(dev), lens.to(dev)
            with torch.autocast(device_type=dev, enabled=False):
                feats, fl = kaldi_fbank_batch(wp, lens)
            fm = torch.arange(feats.shape[1], device=dev)[None, :] >= fl[:, None]
            feats = feats.masked_fill(fm.unsqueeze(-1), 0.0)
            enc, el = model.encode(feats, fl)
            ids = model.ctc_head(enc).argmax(-1).cpu(); el = el.cpu().tolist()
            for k, i in enumerate(bi):
                pred = greedy_ctc_decode(ids[k, :int(el[k])].tolist(), blank_id)
                g = gold[i]
                if args.collapse:
                    def coll(seq):
                        out = []
                        for x in seq:
                            if not out or out[-1] != x: out.append(x)
                        return out
                    g = coll(g); pred = coll(pred)
                e = edit_distance(g, pred)
                tot_err += e; tot_len += len(g); seen += 1
                if e == 0: perfect += 1
                s = rows[i].get("source", "ALL")
                src_err[s] += e; src_len[s] += len(g); src_n[s] += 1; src_perf[s] += (e == 0)
            if seen % 400 < args.enc_batch:
                print(f"  {seen}/{len(idx)} running PER={100*tot_err/max(tot_len,1):.2f}%", flush=True)

    print(f"\n=== Quran PER on {Path(args.manifest).stem} (chunk={_chunk}) ===", flush=True)
    print(f"{'source':20s} {'clips':>5s} {'PER%':>7s} {'exact%':>7s}", flush=True)
    for s in sorted(src_err):
        print(f"{s:20s} {src_n[s]:5d} {100*src_err[s]/max(src_len[s],1):7.2f} {100*src_perf[s]/max(src_n[s],1):7.1f}", flush=True)
    print(f"{'ALL':20s} {seen:5d} {100*tot_err/max(tot_len,1):7.2f} {100*perfect/max(seen,1):7.1f}", flush=True)
    print("QURAN_PER_DONE", flush=True)

if __name__ == "__main__":
    main()
