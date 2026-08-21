"""Cache-aware STREAMING CTC ONNX (sherpa-onnx compatible) for the Quran phoneme zipformer.
Feeds our build_model weights into icefall's export-onnx-streaming-ctc.OnnxModel. We stub the
main()-only imports (train.*, icefall.checkpoint.*, str2bool/num_tokens) so we can import the real
OnnxModel + export_streaming_ctc_model_onnx without running its CLI. Also writes an int8 version. CPU.
"""
import sys, io, os, types, importlib.util
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
import logging; logging.basicConfig(level=logging.ERROR)
sys.path.insert(0, "C:/Users/Anon/research/tarteel-asr/scripts")
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from zipformer_rnnt_ctc_train import import_training_deps, build_model, build_context_profiles, load_tokenizer

ROOT = "C:/Users/Anon/research/tarteel-asr"
ZDIR = ROOT + "/external/icefall/egs/librispeech/ASR/zipformer"
CKPT = ROOT + "/checkpoints/zipformer-phoneme-quran-finetune/quran_finetune_final.pt"
TOK = ROOT + "/data/zipformer_rnnt_ctc/tokenizer/phoneme_units.json"
OUTDIR = ROOT + "/publish/quran-phoneme-zipformer"
OUT = OUTDIR + "/quran_phoneme_zipformer.onnx"
INT8 = OUTDIR + "/quran_phoneme_zipformer.int8.onnx"

import_training_deps(Path(ROOT))
sys.path.insert(0, ZDIR)

# --- stub the imports that export-onnx-streaming-ctc.py only needs for its CLI main() ---
import icefall.utils as iu
if not hasattr(iu, "str2bool"): iu.str2bool = lambda v: str(v).lower() in ("true", "1", "yes", "t")
if not hasattr(iu, "num_tokens"): iu.num_tokens = lambda *a, **k: 0
_ck = types.ModuleType("icefall.checkpoint")
for _n in ["average_checkpoints", "average_checkpoints_with_averaged_model", "find_checkpoints", "load_checkpoint"]:
    setattr(_ck, _n, lambda *a, **k: None)
sys.modules["icefall.checkpoint"] = _ck
_tr = types.ModuleType("train")
_tr.add_model_arguments = lambda *a, **k: None; _tr.get_model = lambda *a, **k: None; _tr.get_params = lambda *a, **k: None
sys.modules["train"] = _tr

spec = importlib.util.spec_from_file_location("exp_stream_ctc", ZDIR + "/export-onnx-streaming-ctc.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
from scaling_converter import convert_scaled_to_non_scaled
print("loaded icefall streaming-ctc exporter", flush=True)

tok = load_tokenizer(TOK); blank = tok.get_piece_size(); vocab = blank + 1
cf = sorted({p.zipformer_chunk_frames for p in build_context_profiles("1000:0.5,640:0.35,320:0.15")})
m = build_model(vocab, blank, cf, 256).eval()
m.encoder.chunk_size = (max(cf),); m.encoder.left_context_frames = (256,)
assert getattr(m.encoder, "causal", False)
m.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
convert_scaled_to_non_scaled(m, inplace=True)
print(f"weights loaded + scaled->non-scaled; vocab={vocab} chunk={max(cf)} left=256", flush=True)

class CtcOut(nn.Module):
    def __init__(self, head): super().__init__(); self.head = head
    def forward(self, x): return F.log_softmax(self.head(x), dim=-1)

onnx_model = mod.OnnxModel(encoder=m.encoder, encoder_embed=m.sub, ctc_output=CtcOut(m.ctc_head))
print(f"chunk_size={onnx_model.chunk_size} left_context_len={onnx_model.left_context_len} pad={onnx_model.pad_length}", flush=True)
mod.export_streaming_ctc_model_onnx(onnx_model, OUT, opset_version=13)
print(f"STREAMING ONNX -> {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)

from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(OUT, INT8, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])
print(f"STREAMING INT8 -> {INT8} ({os.path.getsize(INT8)/1e6:.1f} MB)", flush=True)

import onnxruntime as ort, numpy as np
for f in (OUT, INT8):
    s = ort.InferenceSession(f, providers=["CPUExecutionProvider"])
    meta = s.get_modelmeta().custom_metadata_map
    init = onnx_model.get_init_states()
    feed = {"x": np.random.randn(1, onnx_model.chunk_size * 2 + onnx_model.pad_length, 80).astype(np.float32)}
    si = 0
    for n in [i.name for i in s.get_inputs()]:
        if n == "x": continue
        feed[n] = init[si].numpy(); si += 1
    out = s.run(None, feed)
    print(f"{os.path.basename(f)}: OK log_probs={out[0].shape} model_type={meta.get('model_type')} chunk={meta.get('decode_chunk_len')} T={meta.get('T')}", flush=True)
print("STREAMING_EXPORT_DONE", flush=True)
