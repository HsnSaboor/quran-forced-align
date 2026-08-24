"""Per-word alignment-confidence signals, computed as free post-processing
over data the main Viterbi pass already produces -- no new DP, no new ONNX
inference, no re-running alignment at multiple beam widths.

Two independent signals, combined by `flag_low_confidence_words`:

  - `avg_logprob_along_path` (viterbi.py): absolute acoustic confidence --
    how much probability mass the model puts on the exact chosen phoneme
    sequence. `repeats.py` already computes this, but only for
    repeat-anomaly candidates; this module extends it to every word.
  - decision margin (viterbi.py's `_backtrack_step` return value): a local
    "how close was the runner-up" signal -- the closest analogue an exact
    Viterbi DP has to the disagreement signal a multi-beam-width ensemble
    would give (would a narrower search have gone somewhere else here),
    without materializing a second search.

Both are zero-extra-pass: `log_probs`/`ext`/`path`/`margins` are already
fully resident in memory after `viterbi.ctc_forced_align` returns.
"""
import numpy as np

from .trellis import avg_logprob_along_path


def per_word_min_margin(margins, start_frame, end_frame):
    """Minimum backtrace decision margin over a word's frame span --
    the frame where the chosen path was LEAST decisively better than the
    runner-up alternative. Using min (not mean) deliberately: a single
    genuinely ambiguous frame within a word's span is exactly the kind of
    local uncertainty a beam-disagreement signal is meant to surface, and
    averaging it away with the word's other (likely unambiguous) frames
    would hide it.

    Returns +inf for an empty/invalid span (matching
    `avg_logprob_along_path`'s -inf-for-invalid convention of "fail closed
    so comparisons against it never look falsely confident/unconfident").
    """
    if margins is None or end_frame < start_frame:
        return np.inf
    return float(np.min(margins[start_frame:end_frame + 1]))


def flag_low_confidence_words(cues, log_probs, ext, path, margins,
                               logprob_margin=1.0, decision_margin_floor=0.5):
    """Annotate every cue dict in `cues` (in place, and also returned) with
    `avg_logprob` and `min_decision_margin`, plus a `low_confidence` bool.

    Mirrors the calibration SHAPE `repeats.py`'s anomaly-duration gate
    already uses (a surah-wide baseline, minus a margin) rather than an
    absolute threshold, since "confident" acoustic log-probabilities and
    decision margins are relative to a given surah/reciter/model, not a
    universal constant: a word is flagged low-confidence if EITHER signal
    falls more than its margin below a baseline. `logprob_margin`/
    `decision_margin_floor` are separate thresholds since the two signals
    have unrelated units and scales (log probability nats vs. raw log-prob
    score spread).

    The baseline is the median over MAIN-PASS cues only (`is_repeat` is
    False AND the cue wasn't produced by a repeat splice -- see below),
    deliberately excluding repeat-spliced cues (`is_repeat` True or False
    from `repeats.detect_and_fix_repeats`'s output): those come from a
    small LOCAL doubled-reference re-alignment with different statistical
    properties (shorter windows, forced two-way split) than the
    whole-surah main-pass trellis this function's `log_probs`/`ext`/
    `path`/`margins` arguments describe, so pooling them into one
    surah-wide baseline would conflate two different score distributions.
    Repeat-spliced cues are identified the same way `flag_low_confidence_words`
    itself skips recomputing their signals below: they already carry
    `avg_logprob`/`min_decision_margin` by the time this function runs
    (pre-computed by `repeats.py`'s `_spliced_cue` against their own local
    re-alignment), whereas main-pass cues don't yet.

    This performs one O(T) pass over `margins`/`log_probs` per word (via
    the two per-word helper functions above) -- negligible next to the
    O(T*M) cost of the Viterbi pass that already produced all of this
    module's inputs.
    """
    if not cues:
        return cues

    # Precompute path log-probabilities in one single vectorized GPU/CPU operation
    T = len(path)
    if not isinstance(log_probs, np.ndarray) and hasattr(log_probs, "is_cuda"):
        import torch
        symbols = ext[path]
        symbols_t = torch.as_tensor(symbols, dtype=torch.int64, device=log_probs.device)
        path_logprobs = log_probs[torch.arange(T, device=log_probs.device), symbols_t].cpu().numpy()
    else:
        path_logprobs = log_probs[np.arange(T), ext[path]]
    
    cum_logprobs = np.pad(np.cumsum(path_logprobs), (1, 0))

    is_repeat_spliced = ["avg_logprob" in c for c in cues]

    for c, already_spliced in zip(cues, is_repeat_spliced):
        if already_spliced:
            continue
        sf, ef = c["start_frame"], c["end_frame"]
        if ef >= sf:
            c["avg_logprob"] = float((cum_logprobs[ef + 1] - cum_logprobs[sf]) / (ef - sf + 1))
            c["min_decision_margin"] = per_word_min_margin(margins, sf, ef)
        else:
            c["avg_logprob"] = -np.inf
            c["min_decision_margin"] = np.inf

    baseline_pool = [c for c, spliced in zip(cues, is_repeat_spliced) if not spliced]
    finite_logprobs = [c["avg_logprob"] for c in baseline_pool if np.isfinite(c["avg_logprob"])]
    finite_margins = [c["min_decision_margin"] for c in baseline_pool if np.isfinite(c["min_decision_margin"])]
    logprob_baseline = float(np.median(finite_logprobs)) if finite_logprobs else 0.0
    margin_baseline = float(np.median(finite_margins)) if finite_margins else 0.0

    logprob_floor = logprob_baseline - logprob_margin
    margin_floor = margin_baseline - decision_margin_floor

    for c in cues:
        c["low_confidence"] = c["avg_logprob"] < logprob_floor or c["min_decision_margin"] < margin_floor

    return cues
