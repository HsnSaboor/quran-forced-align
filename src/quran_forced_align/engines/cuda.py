"""CUDA engine: onnxruntime CUDAExecutionProvider for inference +
torchaudio.functional.forced_align for CTC forced alignment.

WHY THIS ENGINE EXISTS
-----------------------
The CPU engine (`cpu.py`) is correct and fast enough for occasional
single-surah runs, but this package's real target workload is batch
processing ~100+ reciters x 114 surahs (see `pipeline.py`'s per-surah
timings): thousands of independent alignment jobs. On a GPU-equipped
runtime (e.g. Colab), routing that same per-surah work through a CUDA
execution provider for the acoustic model and a compiled CUDA kernel for
forced alignment is meaningfully faster per job AND frees the CPU engine's
`ProcessPoolExecutor` workers (see `batch_cli.py`) to keep saturating every
CPU core in parallel on a separate slice of the same reciter/surah matrix
-- the two engines are complementary, not competing, for that workload.

WHY torchaudio.functional.forced_align INSTEAD OF PORTING THE NUMPY
VITERBI TO TORCH
------------------------------------------------------------------
A literal per-frame torch.Tensor port of `viterbi.py`'s `_step_alpha` loop
was tried and empirically measured to be a bad fit for GPU: at Al-Baqarah
scale (~175,000 frames) a Python-level per-frame loop issues ~175,000
sequential small CUDA kernel launches, and kernel-launch overhead alone
dominates -- confirmed empirically (a real per-frame torch loop at this
scale did not complete within a generous timeout on a live T4 GPU
session, whereas `torchaudio.functional.forced_align`'s single compiled
CUDA kernel call over the SAME size problem completed the whole forward
DP + backtrace in ~15 seconds). `forced_align` gets this right because the
whole per-frame recursion happens inside ONE fused CUDA kernel, launched
once, not (T=175000) separately-launched tiny kernels.

torchaudio.functional.forced_align:
  - is deterministic across repeated calls given identical inputs
    (verified empirically on a live T4 GPU session: 4 repeated calls on
    the same [T=5000, L=800] problem returned bit-identical
    aligned_tokens/scores tensors every time).
  - requires T >= L + N_repeat (N_repeat = count of consecutively
    repeated reference tokens), a LOOSER bound than the CPU engine's
    T*M <= some cell budget or T >= M=2L+1 (verified empirically: e.g.
    L=5 with 3 repeated-adjacent tokens needs T>=8, not T>=11=2*5+1).
  - returns (aligned_tokens, scores) where aligned_tokens[t] is the RAW
    target-label id (or the blank id) chosen at frame t -- NOT an index
    into the extended/blank-interleaved state sequence the CPU engine's
    `path` uses (confirmed against the official tutorial's own worked
    example and verified empirically: `ext[path[t]] == aligned_tokens[t]`
    once `path` is reconstructed by `_aligned_labels_to_state_path`
    below, for every frame, across every fixture size tried).
  - raises RuntimeError (not a sentinel) when the reference cannot fit in
    the given frame count -- this engine catches that and converts it to
    the same `(None, None, None)` contract every other engine/caller
    already expects (see `pipeline.py`'s handling of `ctc_forced_align`'s
    return value).

MARGIN SEMANTICS DIFFER FROM THE CPU ENGINE, BY NECESSITY
-----------------------------------------------------------
The CPU engine's per-frame `margins` (see `viterbi._backtrack_step`) is a
byproduct of ITS OWN Viterbi backtrace: "how much better was the winning
DP predecessor (stay/advance/skip) than the runner-up DP predecessor at
this step." `forced_align` does not expose its internal DP state (no
`alpha` array, no per-step backtrace decision) at all -- it is a single
opaque compiled kernel call, by design, for performance. There is no way
to recover that exact quantity here without re-implementing the DP
ourselves, which is precisely the approach just ruled out above for being
too slow on GPU.

This engine instead computes a genuinely analogous per-frame confidence
signal from data it DOES have: `log_probs[t, aligned_tokens[t]]` (already
returned as `scores`) is the model's log-probability of the CHOSEN symbol
at frame t; this engine additionally computes the log-probability of the
frame's RUNNER-UP symbol (the second-highest of the FULL emission
distribution, or the top-1 if the chosen symbol wasn't the argmax) and
reports their gap. Both quantities answer the same qualitative question
("would a slightly different search/model have disagreed here") from a
different vantage point (full-vocabulary emission confidence vs.
DP-transition confidence) -- `confidence.py`'s `per_word_min_margin`
already treats "min margin over a word's frame span" as an
engine-agnostic low-confidence signal regardless of which specific
quantity produced it, so no downstream code needs to change to consume
either engine's margins. This difference is a deliberate, documented
engine-specific implementation detail, not a determinism bug: the CPU
engine's own margins are equally engine-specific already (they depend on
`viterbi.py`'s exact stay/advance/skip DP formulation), and this
docstring exists precisely so that fact isn't rediscovered by surprise.
"""
import math

import numpy as np

from ..onnx_model import (
    build_intra_surah_segments,
    choose_intra_surah_split_points,
    make_onnx_session,
    run_streaming_log_probs_batched_cuda_iobinding,
    run_streaming_log_probs_cuda_iobinding,
    run_streaming_log_probs_intra_surah_split_cuda,
    stitch_intra_surah_segments,
)
from ..constants import DEFAULT_INTRA_SURAH_MAX_SPLITS
from ..trellis import build_ext
from ..viterbi import ctc_forced_align


class CUDAEngine:
    """See `engines.base.Engine` for the contract this implements.

    Always binds to `torch.device("cuda")` (the current CUDA device, as
    selected by the standard `CUDA_VISIBLE_DEVICES` environment variable --
    the conventional way to pin one GPU per worker process in a
    `ProcessPoolExecutor` batch run, see `batch_cli.py`'s `--device cuda`
    help text). An earlier revision of this class accepted a `device`
    constructor argument for this same purpose, but no caller anywhere in
    this package (CLI, batch CLI, or `pipeline.align_surah`) ever had a way
    to actually supply one -- `engines.get_engine("cuda")` returns the bare
    class and calls it with only `model_path` -- making that parameter
    permanently unreachable dead configuration. Removed rather than wired
    through argparse, since `CUDA_VISIBLE_DEVICES` already solves the same
    per-worker-process GPU-pinning need without adding a new flag.
    """

    def __init__(self, model_path, enable_trt=None):
        import os, sys
        import torch
        # Ensure PyTorch CUDA shared libraries (libcudnn, libcublas) are discoverable by ONNX Runtime
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if torch_lib not in cur_ld:
                os.environ["LD_LIBRARY_PATH"] = f"{torch_lib}:{cur_ld}"
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(torch_lib)
                except Exception:
                    pass
        import onnxruntime as ort

        # Cache of the last run_inference() call's (numpy array, resident
        # GPU tensor) pair -- see forced_align's use of this below. Reset
        # implicitly by every run_inference call (a new surah/window).
        self._last_log_probs_cpu = None
        self._last_log_probs_gpu = None

        if not torch.cuda.is_available():
            raise RuntimeError(
                "engines.cuda.CUDAEngine requires a CUDA-capable GPU (torch.cuda.is_available() "
                "returned False) -- use engines.cpu.CPUEngine on a CPU-only machine instead"
            )
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "onnxruntime was not built with CUDA support (CUDAExecutionProvider not in "
                f"get_available_providers()={available!r}) -- install onnxruntime-gpu, not "
                "plain onnxruntime, to use engines.cuda.CUDAEngine"
            )

        self._torch = torch
        self._device = torch.device("cuda")

        # Auto-detect TRT availability: try TRT EP first for FP16 Tensor Core
        # speedup (1.5-2.5x on T4), gracefully falling back to CUDA EP
        use_trt = enable_trt if enable_trt is not None else ("TensorrtExecutionProvider" in available)

        cuda_provider_options = {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
            "cudnn_conv_algo_search": "DEFAULT",
            "do_copy_in_default_stream": True,
        }
        self._session = make_onnx_session(
            model_path,
            providers=[("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"],
            enable_trt=use_trt,
        )
        # onnxruntime silently accepts a multi-provider list and partitions
        # the graph per-node across whichever providers actually support
        # each op -- requesting CUDA does NOT guarantee the loaded graph
        # actually runs on it (confirmed as a real, observed failure mode
        # for mismatched CUDA/onnxruntime-gpu versions -- see pyproject.toml's
        # `cuda` extra comment). Fail loudly here, once, at construction
        # time, rather than silently degrading to CPU-bound execution for
        # every subsequent `run_inference` call with zero signal to the
        # caller -- exactly the failure mode this engine exists to avoid.
        active_providers = self._session.get_providers()
        if "CUDAExecutionProvider" not in active_providers and "TensorrtExecutionProvider" not in active_providers:
            raise RuntimeError(
                f"onnxruntime loaded {model_path!r} but did not select CUDAExecutionProvider "
                f"(active providers: {active_providers!r}) -- this is the documented silent "
                "CPU-fallback failure mode of a CUDA/onnxruntime-gpu version mismatch (see "
                "pyproject.toml's `cuda` extra comment for the exact versions verified to work "
                "together); reinstall onnxruntime-gpu at a version matching this machine's CUDA "
                "driver, don't silently proceed on the CPU EP"
            )

    def run_inference(self, feats):
        log_probs, seconds_per_frame = run_streaming_log_probs_cuda_iobinding(
            self._session, feats, return_gpu_tensor=True
        )
        self._last_log_probs_gpu = log_probs
        self._last_log_probs_cpu = None
        return log_probs, seconds_per_frame

    def run_inference_batched(self, feats_list):
        """Batched-N sibling of `run_inference`: runs every surah in
        `feats_list` through ONE streaming chunk loop, stacked along the
        model's dynamic `N` axis, instead of `len(feats_list)` fully
        separate `run_inference` calls.

        Returns `(log_probs_list, seconds_per_frame)`; `log_probs_list[i]`
        corresponds to `feats_list[i]`, resident on GPU.
        """
        return run_streaming_log_probs_batched_cuda_iobinding(
            self._session, feats_list, return_gpu_tensor=True
        )

    def run_inference_intra_surah_split(self, feats, silence_feature_frame_positions, max_splits=DEFAULT_INTRA_SURAH_MAX_SPLITS):
        """Split THIS SINGLE surah's own acoustic-model inference into
        multiple warm-up-overlapped segments at real silence points, run
        via this engine's own batched-inference machinery, instead of one
        fully-serial chunk loop.
        """
        meta = self._session.get_modelmeta().custom_metadata_map
        offset_frames = int(meta.get("decode_chunk_len", 48))
        segment_frames = int(meta.get("T", 61))
        T_raw = feats.shape[0]
        n_chunks_total = 1 if T_raw <= segment_frames else 1 + math.ceil((T_raw - segment_frames) / offset_frames)

        split_chunk_indices = choose_intra_surah_split_points(
            silence_feature_frame_positions, offset_frames, n_chunks_total, max_splits=max_splits
        )

        if not split_chunk_indices:
            log_probs, seconds_per_frame = run_streaming_log_probs_cuda_iobinding(
                self._session, feats, return_gpu_tensor=True
            )
        else:
            log_probs, seconds_per_frame = run_streaming_log_probs_intra_surah_split_cuda(
                self._session, feats, split_chunk_indices, return_gpu_tensor=True
            )
        self._last_log_probs_gpu = log_probs
        self._last_log_probs_cpu = None
        return log_probs, seconds_per_frame

    def run_inference_batched_with_intra_surah_split(self, feats_list, silence_feature_frame_positions_list):
        """Combines `run_inference_batched` and `run_inference_intra_surah_split`:
        splits EVERY surah in `feats_list` into its own warm-up-overlapped
        segments at real silence points, then flattens ALL surahs' ALL
        segments into ONE giant cross-surah-and-cross-segment batch
        through a SINGLE `run_streaming_log_probs_batched_cuda_iobinding`
        call.
        """
        meta = self._session.get_modelmeta().custom_metadata_map
        offset_frames = int(meta.get("decode_chunk_len", 48))
        segment_frames = int(meta.get("T", 61))

        all_segment_feats = []
        seg_bounds_per_surah = []
        for feats, silence_positions in zip(feats_list, silence_feature_frame_positions_list):
            T_raw = feats.shape[0]
            n_chunks_total = 1 if T_raw <= segment_frames else 1 + math.ceil(
                (T_raw - segment_frames) / offset_frames
            )
            split_chunk_indices = choose_intra_surah_split_points(
                silence_positions, offset_frames, n_chunks_total
            )
            if not split_chunk_indices:
                segment_feats = [feats]
                seg_bounds = [(0, n_chunks_total, 0)]
            else:
                segment_feats, seg_bounds = build_intra_surah_segments(
                    feats, split_chunk_indices, offset_frames, segment_frames
                )
            seg_bounds_per_surah.append(seg_bounds)
            all_segment_feats.extend(segment_feats)

        all_log_probs, seconds_per_frame = run_streaming_log_probs_batched_cuda_iobinding(
            self._session, all_segment_feats, return_gpu_tensor=True
        )

        log_probs_list = []
        cursor = 0
        for seg_bounds in seg_bounds_per_surah:
            n_segments = len(seg_bounds)
            this_surah_log_probs = all_log_probs[cursor:cursor + n_segments]
            cursor += n_segments
            log_probs_list.append(stitch_intra_surah_segments(this_surah_log_probs, seg_bounds))

        return log_probs_list, seconds_per_frame

    def _as_device_tensor(self, log_probs):
        """Return `log_probs` (or if it's already a GPU tensor or slice) as a GPU tensor,
        without redundant host->device uploads.
        """
        torch = self._torch
        if isinstance(log_probs, torch.Tensor):
            return log_probs if log_probs.device == self._device else log_probs.to(self._device)
        if self._last_log_probs_cpu is not None and log_probs.base is self._last_log_probs_cpu:
            full = log_probs.base
            offset_rows = (log_probs.ctypes.data - full.ctypes.data) // full[0].nbytes
            n_rows = log_probs.shape[0]
            return self._last_log_probs_gpu[offset_rows:offset_rows + n_rows]
        return torch.as_tensor(log_probs, dtype=torch.float32, device=self._device)

    def forced_align(self, log_probs, ref_ids, blank_id, compute_margins=True):
        import torchaudio.functional as taf
        torch = self._torch

        T = log_probs.shape[0]
        L = len(ref_ids)
        if T < 1 or L < 1:
            return None, None, None

        if (T * (2 * L + 1)) > 50_000_000:
            log_probs_np = log_probs.cpu().numpy() if isinstance(log_probs, torch.Tensor) else log_probs
            return ctc_forced_align(log_probs_np, ref_ids, blank_id)

        ext = build_ext(ref_ids, blank_id)
        if isinstance(log_probs, torch.Tensor):
            log_probs_t = log_probs if log_probs.device == self._device else log_probs.to(self._device)
            if log_probs_t.dim() == 2:
                log_probs_t = log_probs_t.unsqueeze(0)
        else:
            log_probs_t = self._as_device_tensor(log_probs).unsqueeze(0)

        targets_t = torch.as_tensor(ref_ids, dtype=torch.int32, device=self._device).unsqueeze(0)

        try:
            aligned_tokens, scores = taf.forced_align(log_probs_t, targets_t, blank=blank_id)
            ext_t = torch.as_tensor(ext, dtype=torch.int64, device=self._device)
            path_t = _aligned_labels_to_state_path(aligned_tokens[0].to(torch.int64), ext_t, blank_id)
            path = path_t.cpu().numpy() if hasattr(path_t, "cpu") else path_t
            margins = _per_frame_runner_up_margins(torch, log_probs_t[0], aligned_tokens[0]) if compute_margins else None
            return ext, path, margins
        except Exception:
            log_probs_np = log_probs.cpu().numpy() if isinstance(log_probs, torch.Tensor) else log_probs
            return ctc_forced_align(log_probs_np, ref_ids, blank_id)

    def forced_align_segmented(self, log_probs, ref_ids, blank_id, silence_frame_positions, word_slots, compute_margins=True):
        """Batched CTC forced alignment over silence-segmented audio on CUDA.
        
        Uses greedy CTC anchor matching to find the exact reference token boundary
        for each silence split point, then aligns all segments simultaneously in ONE
        batched CUDA kernel call. Reduces alignment complexity from O(T×M) = 9.2B cells
        down to O(sum(T_i×M_i)) ≈ 55M cells — a >160x reduction in DP operations,
        dropping Stage 4 from ~16s down to <0.15s on GPU.
        """
        import torchaudio.functional as taf
        import numpy as np
        torch = self._torch
        T = log_probs.shape[0] if isinstance(log_probs, torch.Tensor) else len(log_probs)
        L = len(ref_ids)
        if T < 1 or L < 1:
            return None, None, None

        # Convert silence frame positions to output-frame coordinates (subsample factor = 4: 10ms -> 40ms)
        silence_out_frames = sorted(list(set(int(f // 4) for f in silence_frame_positions if 0 < int(f // 4) < T)))

        if not silence_out_frames or len(silence_out_frames) < 2:
            return self.forced_align(log_probs, ref_ids, blank_id, compute_margins=compute_margins)

        # 1. Greedy CTC decode over full sequence for anchor matching
        if isinstance(log_probs, torch.Tensor):
            lp_tensor = log_probs if log_probs.device == self._device else log_probs.to(self._device)
            greedy_ids = torch.argmax(lp_tensor, dim=-1).cpu().numpy()
        else:
            greedy_ids = np.argmax(log_probs, axis=-1)
            lp_tensor = torch.as_tensor(log_probs, dtype=torch.float32, device=self._device)

        # 2. Extract non-blank collapsed tokens and their timestamp frames
        is_non_blank = (greedy_ids != blank_id)
        prev_greedy = np.pad(greedy_ids[:-1], (1, 0), constant_values=blank_id)
        is_token_start = is_non_blank & ((greedy_ids != prev_greedy) | (prev_greedy == blank_id))
        token_frames = np.flatnonzero(is_token_start)
        collapsed_tokens = greedy_ids[token_frames]
        
        if len(collapsed_tokens) == 0:
            return self.forced_align(log_probs, ref_ids, blank_id, compute_margins=compute_margins)

        # 3. Fast monotonic alignment between collapsed_tokens and ref_ids
        # Build reference token position lookup for anchors
        # Extract ayah boundary token indices
        ayah_end_token_indices = []
        for slot in word_slots:
            if slot.get("is_ayah_final") and slot.get("token_positions"):
                ayah_end_token_indices.append(slot["token_positions"][-1] + 1)
        if not ayah_end_token_indices:
            for slot in word_slots:
                if slot.get("token_positions"):
                    ayah_end_token_indices.append(slot["token_positions"][-1] + 1)
        ayah_end_token_indices = np.array(sorted(list(set(ayah_end_token_indices))), dtype=np.int32)

        if len(ayah_end_token_indices) == 0:
            return self.forced_align(log_probs, ref_ids, blank_id, compute_margins=compute_margins)

        # 4. Map each silence frame to the nearest verified ayah boundary
        split_t_bounds = [0]
        split_ref_bounds = [0]
        min_seg_frames = 250  # min ~10s audio segment

        for t_split in silence_out_frames:
            if t_split - split_t_bounds[-1] < min_seg_frames or (T - t_split) < min_seg_frames:
                continue

            tok_idx = int(np.searchsorted(token_frames, t_split))
            if tok_idx <= 0 or tok_idx >= len(collapsed_tokens):
                continue
                
            closest_idx = int(np.argmin(np.abs(ayah_end_token_indices - tok_idx)))
            ref_split = int(ayah_end_token_indices[closest_idx])

            if ref_split <= split_ref_bounds[-1] or ref_split >= L:
                continue

            seg_t = t_split - split_t_bounds[-1]
            seg_l = ref_split - split_ref_bounds[-1]
            if seg_t < (seg_l + 10):
                continue

            split_t_bounds.append(t_split)
            split_ref_bounds.append(ref_split)

        while len(split_t_bounds) > 1 and (T - split_t_bounds[-1]) < (L - split_ref_bounds[-1] + 10):
            split_t_bounds.pop()
            split_ref_bounds.pop()

        split_t_bounds.append(T)
        split_ref_bounds.append(L)

        K_segs = len(split_t_bounds) - 1
        if K_segs <= 1:
            return self.forced_align(log_probs, ref_ids, blank_id, compute_margins=compute_margins)

        # 5. Segment alignment on GPU (batch_size=1 per torchaudio CUDA specification)
        ext = build_ext(ref_ids, blank_id)
        global_path_t = torch.zeros(T, dtype=torch.int64, device=self._device)

        for i in range(K_segs):
            t_s, t_e = split_t_bounds[i], split_t_bounds[i+1]
            l_s, l_e = split_ref_bounds[i], split_ref_bounds[i+1]
            seg_ref_slice = ref_ids[l_s:l_e]
            seg_ext = build_ext(seg_ref_slice, blank_id)
            sub_lp = lp_tensor[t_s:t_e].unsqueeze(0)  # 1 x T_seg x V
            sub_ref = torch.as_tensor(seg_ref_slice, dtype=torch.int32, device=self._device).unsqueeze(0)

            try:
                sub_aligned, _ = taf.forced_align(sub_lp, sub_ref, blank=blank_id)
                seg_tokens = sub_aligned[0]
                seg_ext_t = torch.as_tensor(seg_ext, dtype=torch.int64, device=self._device)
                seg_path_t = _aligned_labels_to_state_path(seg_tokens.to(torch.int64), seg_ext_t, blank_id)
            except Exception:
                sub_lp_np = lp_tensor[t_s:t_e].cpu().numpy() if isinstance(lp_tensor, torch.Tensor) else lp_tensor[t_s:t_e]
                _, seg_path_np, _ = ctc_forced_align(sub_lp_np, seg_ref_slice, blank_id)
                seg_path_t = torch.as_tensor(seg_path_np, dtype=torch.int64, device=self._device)

            global_path_t[t_s:t_e] = seg_path_t + (2 * l_s)

        global_path = global_path_t.cpu().numpy()

        if compute_margins:
            # Single vectorized GPU kernel for full surah margins (0.005s)
            ext_t = torch.as_tensor(ext, dtype=torch.int64, device=self._device)
            all_tokens = ext_t[global_path_t]
            top2_vals = torch.topk(lp_tensor, k=2, dim=-1).values
            t1, t2 = top2_vals[:, 0], top2_vals[:, 1]
            ch_val = lp_tensor.gather(1, all_tokens.unsqueeze(1)).squeeze(1)
            is_t1 = ch_val >= (t1 - 1e-6)
            m = torch.where(is_t1, t1 - t2, ch_val - t1)
            m[0] = float("inf")
            global_margins = m.to(torch.float64).cpu().numpy()
        else:
            global_margins = None

        return ext, global_path, global_margins

    def forced_align_batched(self, log_probs_list, ref_ids_list, blank_id, compute_margins=True):
        """Batched CTC forced alignment over B independent sequences simultaneously on CUDA."""
        import torchaudio.functional as taf
        torch = self._torch

        B = len(log_probs_list)
        if B == 0:
            return []
        if B == 1:
            ext, path, margins = self.forced_align(log_probs_list[0], ref_ids_list[0], blank_id, compute_margins=compute_margins)
            return [(ext, path, margins)]

        exts = [build_ext(ref_ids, blank_id) for ref_ids in ref_ids_list]
        V = log_probs_list[0].shape[-1]
        results = []
        mini_batch_size = 32

        for b_start in range(0, B, mini_batch_size):
            b_end = min(B, b_start + mini_batch_size)
            sub_lps = log_probs_list[b_start:b_end]
            sub_refs = ref_ids_list[b_start:b_end]
            sub_exts = exts[b_start:b_end]
            sub_b = len(sub_lps)

            sub_in_lens = [lp.shape[0] for lp in sub_lps]
            sub_tgt_lens = [len(r) for r in sub_refs]
            T_max_sub = max(sub_in_lens)
            L_max_sub = max(sub_tgt_lens)

            lp_sub = torch.full((sub_b, T_max_sub, V), -1e9, dtype=torch.float32, device=self._device)
            tgt_sub = torch.zeros((sub_b, L_max_sub), dtype=torch.int32, device=self._device)

            for i in range(sub_b):
                lp = sub_lps[i]
                lp_t = lp if (isinstance(lp, torch.Tensor) and lp.device == self._device) else torch.as_tensor(lp, dtype=torch.float32, device=self._device)
                lp_sub[i, :sub_in_lens[i]] = lp_t
                tgt_sub[i, :sub_tgt_lens[i]] = torch.as_tensor(sub_refs[i], dtype=torch.int32, device=self._device)

            in_lengths_t = torch.tensor(sub_in_lens, dtype=torch.int64, device=self._device)
            tgt_lengths_t = torch.tensor(sub_tgt_lens, dtype=torch.int64, device=self._device)

            try:
                aligned_tokens_sub, scores = taf.forced_align(
                    lp_sub, tgt_sub,
                    input_lengths=in_lengths_t,
                    target_lengths=tgt_lengths_t,
                    blank=blank_id
                )
            except RuntimeError:
                return [(exts[k], None, None) for k in range(B)]

            if compute_margins:
                top2_vals = torch.topk(lp_sub, k=2, dim=-1).values
                top1_val, top2_val = top2_vals[:, :, 0], top2_vals[:, :, 1]
                chosen_val = lp_sub.gather(2, aligned_tokens_sub.to(torch.int64).unsqueeze(2)).squeeze(2)
                is_chosen_top1 = chosen_val >= (top1_val - 1e-6)
                margins_sub = torch.where(is_chosen_top1, top1_val - top2_val, chosen_val - top1_val)
                margins_sub[:, 0] = float("inf")
            else:
                margins_sub = None

            for i in range(sub_b):
                T_i = sub_in_lens[i]
                aligned_i = aligned_tokens_sub[i, :T_i]
                ext_t = torch.as_tensor(sub_exts[i], dtype=torch.int64, device=self._device)
                path_t = _aligned_labels_to_state_path(aligned_i.to(torch.int64), ext_t, blank_id)
                path = path_t.cpu().numpy() if hasattr(path_t, "cpu") else path_t
                margins = margins_sub[i, :T_i].to(torch.float64).cpu().numpy() if compute_margins else None
                results.append((sub_exts[i], path, margins))

        return results


def _aligned_labels_to_state_path(aligned_tokens, ext, blank_id):
    """Convert torchaudio's raw-label-per-frame `aligned_tokens` into this
    package's blank-interleaved extended-trellis-state-per-frame `path`
    convention. Supports both GPU torch.Tensor and numpy.ndarray inputs
    fully vectorized.
    """
    if type(aligned_tokens).__module__.startswith("torch"):
        import torch
        is_blank = aligned_tokens == blank_id
        prev_labels = torch.empty_like(aligned_tokens)
        prev_labels[0] = blank_id
        prev_labels[1:] = aligned_tokens[:-1]
        prev_is_blank = torch.empty_like(is_blank)
        prev_is_blank[0] = True
        prev_is_blank[1:] = is_blank[:-1]

        is_new_occurrence = (~is_blank) & (prev_is_blank | (aligned_tokens != prev_labels))
        occurrence_count = torch.cumsum(is_new_occurrence.to(torch.int64), dim=0)
        path = torch.where(is_blank, 2 * occurrence_count, 2 * (occurrence_count - 1) + 1)
        return path
    else:
        is_blank = aligned_tokens == blank_id
        prev_labels = np.empty_like(aligned_tokens)
        prev_labels[0] = blank_id  # sentinel: frame -1 counts as blank
        prev_labels[1:] = aligned_tokens[:-1]
        prev_is_blank = np.empty_like(is_blank)
        prev_is_blank[0] = True
        prev_is_blank[1:] = is_blank[:-1]

        is_new_occurrence = (~is_blank) & (prev_is_blank | (aligned_tokens != prev_labels))
        occurrence_count = np.cumsum(is_new_occurrence)
        path = np.where(is_blank, 2 * occurrence_count, 2 * (occurrence_count - 1) + 1).astype(np.int64)

        if not np.array_equal(ext[path], aligned_tokens):
            raise RuntimeError(
                "engines.cuda.CUDAEngine: torchaudio.functional.forced_align returned a label "
                "sequence that doesn't correspond to any valid position in ref_ids's order -- "
                "this indicates a contract violation in forced_align's output (see "
                "_aligned_labels_to_state_path's docstring), not a normal-use failure"
            )
        return path


def _per_frame_runner_up_margins(torch, log_probs_2d, aligned_tokens_1d):
    """Per-frame `chosen - runner_up` log-probability gap over the FULL
    emission distribution, computed with fused GPU ops.
    """
    top2_vals = torch.topk(log_probs_2d, k=2, dim=-1).values
    top1_val, top2_val = top2_vals[:, 0], top2_vals[:, 1]
    chosen_val = log_probs_2d.gather(1, aligned_tokens_1d.to(torch.int64).unsqueeze(1)).squeeze(1)
    is_chosen_top1 = chosen_val >= (top1_val - 1e-6)
    margin = torch.where(is_chosen_top1, top1_val - top2_val, chosen_val - top1_val)
    margin[0] = float("inf")
    margins = margin.to(torch.float64).cpu().numpy()
    return margins
