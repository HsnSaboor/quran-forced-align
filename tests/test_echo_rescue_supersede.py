"""Rescue supersede-drop: a successful echo-rescue replaces the anchor's
broad K=1 copy2 (it spans both echoes unsplit) instead of stacking a
second, overlapping copy2 of the same word.

Strategy: stub the C-engine loader (forced K=1 accept at the anchor with
a known broad copy2 span) and the rescue (success with two narrower
copy2 spans). This isolates the caller-side drop contract in
`_detect_and_fix_repeats_pass` deterministically, with no audio and no
dependence on real acoustic gates (covered separately by the 6:156
production probes).
"""
import ctypes
import os
import sys

sys.path.insert(0, "src")

# Rescue is default-ON; clear any kill switch for this module's tests.
os.environ.pop("QFA_DISABLE_ECHO_RESCUE", None)

import numpy as np

import quran_forced_align.repeats.detection as det


def _cue(word_idx, s, e, ntoks=3):
    return {
        "word": f"w{word_idx}", "sura": 6, "aya": 156, "word_idx": word_idx,
        "is_ayah_final": False, "start_frame": s, "end_frame": e,
        "is_repeat": False, "token_positions": list(range(ntoks)),
        "token_char_idx": [], "letters": [],
    }


_WS = 5  # fake window start for the anchor's accepted K=1


class _FakeEngine:
    """Reports K=1 accepted at cue 1; path walks states 0..12 then parks."""

    def fast_detect_and_fix_repeats_engine(self, *a):
        out_K = ctypes.cast(a[-6], ctypes.POINTER(ctypes.c_int32))
        out_ws = ctypes.cast(a[-5], ctypes.POINTER(ctypes.c_int32))
        out_we = ctypes.cast(a[-4], ctypes.POINTER(ctypes.c_int32))
        out_off = ctypes.cast(a[-3], ctypes.POINTER(ctypes.c_int32))
        out_ln = ctypes.cast(a[-2], ctypes.POINTER(ctypes.c_int32))
        out_paths = ctypes.cast(a[-1], ctypes.POINTER(ctypes.c_int32))
        out_K[1] = 1
        out_ws[1] = _WS
        out_we[1] = 69
        out_off[1] = 0
        out_ln[1] = 65
        for t in range(65):
            out_paths[t] = t if t < 13 else 12
        return 0


def _fake_rescue(anchor_i, accepted_words, cues, combined, blank,
                 min_dur, floor, lp, spf, tfs_fn, splice_fn,
                 fixed, consumed, samples):
    assert anchor_i == 1 and list(accepted_words) == [1]
    # Two narrower K=2 copy2 spans (same contract as the real rescue).
    for j, (s, e) in ((0, (50, 55)), (1, (56, 61))):
        c = dict(cues[j])
        c["start_frame"], c["end_frame"] = s, e
        c["is_repeat"] = True
        c["repeat_stage"] = "stage1-echo-rescue"
        fixed.append(c)
        consumed.add(j)
    fixed.sort(key=lambda c: (c["start_frame"], c["end_frame"]))
    return True


def test_rescue_success_supersedes_broad_k1_copy2(monkeypatch):
    from quran_forced_align import decode as dec_mod
    import quran_forced_align.repeats.echo_rescue as er
    monkeypatch.setattr(dec_mod, "get_fast_ops", lambda: _FakeEngine())
    monkeypatch.setattr(er, "rescue_decayed_echo_left", _fake_rescue)

    # Anchor cue 1 with 3 tokens; the fake path walks states 0,1,2...
    # so the staged copy2 (doubled-L=3 states 7..11) lands at local
    # [7, 11] -> absolute [WS+7, WS+11] = [12, 16].
    cues = [_cue(10, 0, 9), _cue(11, 10, 30), _cue(12, 70, 79)]
    T, V, blank = 90, 4, 3
    lp = np.full((T, V), -3.0, dtype=np.float64)
    lp[:, 0] = -0.01
    combined = [0] * 9
    ext = np.array([blank, 0] * 9 + [blank], dtype=np.int64)
    path = np.zeros(T, dtype=np.int64)
    samples = np.zeros(T * 640, dtype=np.float32)

    out = det._detect_and_fix_repeats_pass(
        None, [dict(c) for c in cues], lp, combined, blank, ext, path,
        0.15, 3.0, 2, seconds_per_frame=0.04, samples=samples,
    )
    reps = [c for c in out if c.get("is_repeat")]
    for c in reps:
        print("REP", c["word_idx"], c["start_frame"], c["end_frame"],
              c.get("repeat_stage"))
    # Rescue fired -> exactly its two spans survive; the staged broad
    # K=1 copy2 is dropped (no same-word overlap allowed regardless).
    assert any(c.get("repeat_stage") == "stage1-echo-rescue" for c in reps)
    assert not any(c.get("repeat_stage") == "c-engine" for c in reps)
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            a, b = reps[i], reps[j]
            if a.get("word_idx") == b.get("word_idx"):
                assert a["end_frame"] < b["start_frame"] or \
                    b["end_frame"] < a["start_frame"], (a, b)


def test_rescue_failure_keeps_broad_k1_copy2(monkeypatch):
    """Refusal path: no rescue success -> staged K=1 copy2 untouched."""
    from quran_forced_align import decode as dec_mod
    import quran_forced_align.repeats.echo_rescue as er
    monkeypatch.setattr(dec_mod, "get_fast_ops", lambda: _FakeEngine())
    monkeypatch.setattr(er, "rescue_decayed_echo_left", lambda *a, **k: False)

    cues = [_cue(10, 0, 9), _cue(11, 10, 30), _cue(12, 70, 79)]
    T, V, blank = 90, 4, 3
    lp = np.full((T, V), -3.0, dtype=np.float64)
    lp[:, 0] = -0.01
    combined = [0] * 9
    ext = np.array([blank, 0] * 9 + [blank], dtype=np.int64)
    path = np.zeros(T, dtype=np.int64)
    samples = np.zeros(T * 640, dtype=np.float32)

    out = det._detect_and_fix_repeats_pass(
        None, [dict(c) for c in cues], lp, combined, blank, ext, path,
        0.15, 3.0, 2, seconds_per_frame=0.04, samples=samples,
    )
    reps = [c for c in out if c.get("is_repeat")]
    assert len(reps) == 1 and reps[0].get("repeat_stage") == "c-engine"
    assert (reps[0]["start_frame"], reps[0]["end_frame"]) == (_WS + 7, _WS + 11)
