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
import numpy as np

from ..onnx_model import make_onnx_session, run_streaming_log_probs
from ..trellis import build_ext


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

    def __init__(self, model_path):
        # torch/torchaudio/onnxruntime's GPU build are optional dependencies
        # (see pyproject.toml's `cuda` extra) -- importing them lazily here,
        # not at module load time, means every CPU-only caller (the
        # default engine, and this whole package's test suite on a
        # CPU-only machine) never needs torch/torchaudio/onnxruntime-gpu
        # installed at all, matching `engines.cpu.CPUEngine`'s zero-torch
        # footprint.
        import onnxruntime as ort
        import torch

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
        self._session = make_onnx_session(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
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
        if "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(
                f"onnxruntime loaded {model_path!r} but did not select CUDAExecutionProvider "
                f"(active providers: {active_providers!r}) -- this is the documented silent "
                "CPU-fallback failure mode of a CUDA/onnxruntime-gpu version mismatch (see "
                "pyproject.toml's `cuda` extra comment for the exact versions verified to work "
                "together); reinstall onnxruntime-gpu at a version matching this machine's CUDA "
                "driver, don't silently proceed on the CPU EP"
            )

    def run_inference(self, feats):
        # float32, not this function's float64 default: torchaudio's
        # forced_align (called below) only accepts float32 log-probs
        # anyway, so requesting float64 here first would force an
        # avoidable full-matrix float32->float64->float32 round trip for
        # data whose extra precision gets discarded again immediately --
        # see run_streaming_log_probs's `output_dtype` docstring.
        return run_streaming_log_probs(self._session, feats, output_dtype=np.float32)

    def forced_align(self, log_probs, ref_ids, blank_id):
        import torchaudio.functional as taf
        torch = self._torch

        T = log_probs.shape[0]
        L = len(ref_ids)
        if T < 1 or L < 1:
            return None, None, None

        ext = build_ext(ref_ids, blank_id)
        # log_probs is already float32 (see run_inference above) for every
        # whole-surah call; repeats/candidate.py's local re-alignment
        # windows are plain numpy SLICES of that same float32 array, so
        # this is a no-op dtype-cast (copy=False-equivalent via
        # as_tensor's own zero-copy-when-possible behavior) in the common
        # case, not a redundant cast.
        log_probs_t = torch.as_tensor(log_probs, dtype=torch.float32, device=self._device).unsqueeze(0)
        targets_t = torch.as_tensor(ref_ids, dtype=torch.int32, device=self._device).unsqueeze(0)

        try:
            aligned_tokens, scores = taf.forced_align(log_probs_t, targets_t, blank=blank_id)
        except RuntimeError:
            # torchaudio raises (rather than returning a sentinel) when the
            # reference cannot fit in this many frames -- normalize to this
            # module's documented (None, None, None) contract, matching
            # every other engine/caller (see this module's docstring).
            return None, None, None

        aligned_tokens_np = aligned_tokens[0].to(torch.int64).cpu().numpy()
        path = _aligned_labels_to_state_path(aligned_tokens_np, ext, blank_id)
        margins = _per_frame_runner_up_margins(torch, log_probs_t[0], aligned_tokens[0])
        return ext, path, margins


def _aligned_labels_to_state_path(aligned_tokens, ext, blank_id):
    """Convert torchaudio's raw-label-per-frame `aligned_tokens` into this
    package's blank-interleaved extended-trellis-state-per-frame `path`
    convention (state 2*p is the blank before ref_ids[p], state 2*p+1 is
    the label state for ref_ids[p]) -- fully vectorized (no Python loop
    over T), which matters since T is up to ~175,000 for the largest
    surah. `ext` (the blank-interleaved extended state sequence this
    package's other engine also builds via `trellis.build_ext`) is passed
    in rather than rebuilt from `ref_ids` here, since the caller
    (`CUDAEngine.forced_align`) already has it and every consumer of this
    function's output needs the SAME `ext` array for its own `ext[path]`
    lookups anyway.

    Exploits the same rule torchaudio's own tutorial documents for
    telling repeated-adjacent labels apart (see this module's docstring
    and `torchaudio.functional.merge_tokens`'s documented behaviour): a
    frame starts a NEW occurrence of a label iff that label differs from
    blank AND either the previous frame was blank or the previous frame's
    label was different. The running count of "new occurrence starts" IS
    the 0-indexed position into `ref_ids` this frame's label state
    corresponds to; a blank frame's state is twice that running count
    (the blank immediately before the NEXT not-yet-started label), and a
    label frame's state is twice (running_count - 1) + 1 (the label state
    for the occurrence that just started or is continuing).

    Verified empirically (on a live T4 GPU session, across multiple
    problem sizes up to Al-Baqarah scale) that `build_ext(ref_ids,
    blank_id)[path] == aligned_tokens` for every frame under this
    reconstruction, and that the resulting `path` is monotonic
    non-decreasing (a structural requirement every downstream consumer,
    e.g. `trellis.frame_spans_from_path`, already assumes). That
    equivalence is re-checked below on every real call (not just in
    tests): unlike the CPU engine's backtrace (an algebraic identity
    re-derived from the DP's own recorded `alpha` array, which cannot
    diverge from what the forward pass actually computed), this is a
    post-hoc reconstruction with no closed-loop guarantee of its own --
    if `torchaudio.functional.forced_align` ever returned labels out of
    `ref_ids`'s order (a contract violation on its part, not expected in
    normal use, but not something this function can rule out on its own),
    this reconstruction would silently produce a state path that doesn't
    correspond to `aligned_tokens` at all. Failing loudly here costs one
    O(T) array comparison, negligible next to the O(T) work already done
    to build `path`.
    """
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
    emission distribution, computed once for every frame with two fused
    GPU ops (`torch.topk` + a `where`) -- see this module's docstring
    ("MARGIN SEMANTICS DIFFER...") for why this is the CUDA engine's
    analogue of the CPU engine's DP-backtrace margin, not the same
    quantity.

    `margins[0]` is set to +inf, matching every engine's "no meaningful
    margin at the very first frame" convention (there is no frame -1 to
    have disagreed with).
    """
    top2_vals = torch.topk(log_probs_2d, k=2, dim=-1).values
    top1_val, top2_val = top2_vals[:, 0], top2_vals[:, 1]
    chosen_val = log_probs_2d.gather(1, aligned_tokens_1d.to(torch.int64).unsqueeze(1)).squeeze(1)
    is_chosen_top1 = torch.isclose(chosen_val, top1_val)
    margin = torch.where(is_chosen_top1, top1_val - top2_val, chosen_val - top1_val)
    margins = margin.to(torch.float64).cpu().numpy()
    margins[0] = np.inf
    return margins
