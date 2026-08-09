"""Engine protocol: the contract every forced-alignment execution engine
(`cpu.py`, `cuda.py`) implements, so `pipeline.py` and every other
consumer can call either one identically."""
from typing import Protocol


class Engine(Protocol):
    """A forced-alignment execution engine bound to one loaded acoustic
    model. Construct via `engines.get_engine(name)(model_path)`.
    """

    def run_inference(self, feats):
        """Run the streaming acoustic model over `feats` ([T_raw, 80]
        float32 fbank features) and return `(log_probs, seconds_per_frame)`
        -- `log_probs` is a [T, V] float64 array (V = vocab size including
        blank), `seconds_per_frame` is the real-time duration one output
        frame of `log_probs` spans."""
        ...

    def forced_align(self, log_probs, ref_ids, blank_id):
        """CTC forced-alignment Viterbi over the blank-interleaved
        extended state sequence built from `ref_ids`. Returns
        `(ext, path, margins)`:

          - `ext`: the blank-interleaved extended state sequence
            [blank, ref_ids[0], blank, ref_ids[1], blank, ...] (int64
            array, length M = 2*len(ref_ids)+1).
          - `path`: the best extended-trellis state at each frame of
            `log_probs` (int64 array, length T, monotonic non-decreasing).
          - `margins`: a per-frame alignment-confidence signal (float64
            array, length T; `margins[0]` is always +inf, matching every
            engine's "no backtrace step into frame 0" convention).

        Returns `(None, None, None)` if `ref_ids` cannot possibly fit in
        `log_probs`'s frame count (too few frames for the reference),
        exactly like every existing caller (`pipeline.py`, `repeats.py`)
        already expects and handles.
        """
        ...
