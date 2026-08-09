"""Unit tests for engines.cuda's pure-numpy state-path reconstruction
(`_aligned_labels_to_state_path`) -- this logic has no GPU or torch
dependency at all (it operates on plain numpy arrays representing what
torchaudio.functional.forced_align WOULD have returned), so it is fully
unit-testable without a CUDA-capable machine or even torch installed,
closing a coverage gap found in code review ("no test at all exercises
engines/cuda.py").

Does NOT import quran_forced_align.engines.cuda as a module-level import
(that module imports onnxruntime at module scope for its `CUDAEngine`
class, and torch/torchaudio lazily inside methods) -- importing it here
only requires onnxruntime, which every install already has one flavor of
(see pyproject.toml's cpu/cuda extras), so this test file runs in both.
"""
import numpy as np
import pytest

from quran_forced_align.engines.cuda import _aligned_labels_to_state_path
from quran_forced_align.trellis import build_ext


def _check(aligned_tokens, ref_ids, blank_id):
    aligned_tokens = np.asarray(aligned_tokens, dtype=np.int64)
    ext = build_ext(ref_ids, blank_id)
    path = _aligned_labels_to_state_path(aligned_tokens, ext, blank_id)
    assert np.array_equal(ext[path], aligned_tokens), "ext[path] must reproduce aligned_tokens exactly"
    assert np.all(np.diff(path) >= 0), "path must be monotonic non-decreasing"
    return path


def test_simple_two_label_sequence_with_repeats():
    # ref_ids = [5, 7]; each label occupies multiple consecutive frames,
    # separated by one blank frame (mirrors a real streaming CTC output).
    path = _check([5, 5, 0, 7, 7, 7], ref_ids=[5, 7], blank_id=0)
    assert path.tolist() == [1, 1, 2, 3, 3, 3]


def test_leading_label_with_no_preceding_blank_frame():
    # First frame is already a real label (not blank) -- the "prev frame
    # was blank" sentinel must treat frame -1 as blank so this is still
    # correctly recognized as a NEW occurrence, not silently skipped.
    # ext = [0, 5, 0, 7, 0]; frame0="5" starts occurrence 0 (state 1),
    # frame1=blank sits between the two occurrences (state 2, the blank
    # immediately before ref_ids[1]=7 -- not state 0, which was already
    # passed and can never be revisited since path is monotonic),
    # frame2="7" starts occurrence 1 (state 3).
    path = _check([5, 0, 7], ref_ids=[5, 7], blank_id=0)
    assert path.tolist() == [1, 2, 3]


def test_adjacent_identical_labels_separated_by_blank_are_two_occurrences():
    # "a - a" (two occurrences of the SAME label, separated by a real
    # blank frame) must produce TWO distinct label states, not be
    # collapsed into one continuous occurrence -- this is exactly the
    # "aabb"-style disambiguation torchaudio's own docs describe.
    path = _check([5, 0, 5], ref_ids=[5, 5], blank_id=0)
    # ext = [0, 5, 0, 5, 0]; first "5" at state 1, blank at state 2, second "5" at state 3.
    assert path.tolist() == [1, 2, 3]


def test_adjacent_identical_labels_with_no_blank_stay_one_occurrence():
    # "a a" with NO intervening blank frame must be treated as ONE
    # continuing occurrence (the label state doesn't advance) -- this
    # matches standard CTC collapsing semantics.
    path = _check([5, 5], ref_ids=[5], blank_id=0)
    assert path.tolist() == [1, 1]


def test_all_blank_frames_before_any_label():
    path = _check([0, 0, 5], ref_ids=[5], blank_id=0)
    assert path.tolist() == [0, 0, 1]


def test_trailing_blank_frames_after_last_label():
    path = _check([5, 0, 0], ref_ids=[5], blank_id=0)
    assert path.tolist() == [1, 2, 2]


def test_out_of_order_labels_raise_instead_of_silently_producing_garbage():
    # Adversarial input: aligned_tokens claims ref_ids[1]=7 appears before
    # ref_ids[0]=5 ever does -- a contract violation forced_align itself
    # should never produce in normal use, but this function must fail
    # loudly rather than silently return a path that doesn't correspond
    # to aligned_tokens at all (see the function's own docstring on this).
    aligned_tokens = np.asarray([7, 7, 0], dtype=np.int64)
    ext = build_ext([5, 7], blank_id=0)
    with pytest.raises(RuntimeError, match="forced_align returned a label sequence"):
        _aligned_labels_to_state_path(aligned_tokens, ext, blank_id=0)
