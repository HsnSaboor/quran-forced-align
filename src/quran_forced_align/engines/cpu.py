"""CPU engine: raw-ONNX CPUExecutionProvider + hand-rolled numpy Viterbi.

Thin `Engine`-protocol wrapper around the pre-existing, unmodified
`onnx_model.py`/`viterbi.py` implementation -- this is the same code path
every previous release of this package used, byte-for-byte; wrapping it
in this class changes no numerics, only how `pipeline.py` selects it.
"""
from ..onnx_model import make_onnx_session, run_streaming_log_probs
from ..viterbi import ctc_forced_align


class CPUEngine:
    """See `engines.base.Engine` for the contract this implements."""

    def __init__(self, model_path):
        self._session = make_onnx_session(model_path)

    def run_inference(self, feats):
        return run_streaming_log_probs(self._session, feats)

    def forced_align(self, log_probs, ref_ids, blank_id, compute_margins=True):
        return ctc_forced_align(log_probs, ref_ids, blank_id)
