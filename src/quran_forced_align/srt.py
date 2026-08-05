"""SRT/JSON emission: timestamp formatting, SRT writer, frame-cue -> tuple
conversion, JSON writer.

`fmt_srt_time` and `emit_srt` re-implement the small, generic SRT-writer
utilities that lived in the older build_surah_srt.py script (that script's
own alignment logic is dead weight and intentionally left behind -- only
this generic formatting/writing logic is reused here).
"""
import json

from .constants import MIN_WORD_DUR


def fmt_srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    mi, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{mi:02d}:{s:02d},{ms:03d}"


def emit_srt(cues, out_path):
    lines = []
    for idx, (word, start, end, sura, aya, is_repeat) in enumerate(cues, 1):
        tag = " [repeat]" if is_repeat else ""
        lines.append(f"{idx}\n{fmt_srt_time(start)} --> {fmt_srt_time(end)}\n{word}{tag}\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path} ({len(cues)} word cues, "
          f"{sum(1 for c in cues if c[5])} flagged as repeats)")


def cues_to_tuples(cues, seconds_per_frame):
    """Convert frame-indexed cues to (word, start, end, sura, aya, is_repeat)
    tuples, sorted by start time.

    Raw frame-derived spans from the Viterbi path are monotonic and
    non-overlapping by construction (each frame is assigned to exactly one
    trellis state, and states map 1:1 to word-owned token positions in
    strictly increasing order). The MIN_WORD_DUR floor below (applied to
    pad degenerate near-zero-length cues, e.g. a word compressed into a
    single frame) can push a word's `end` PAST the next word's `start` if
    applied naively -- confirmed empirically on the Al-Fatiha smoke test
    (6/34 cues overlapped before this clamp was added). Fix: after sorting,
    clamp each cue's floored end to the next cue's start so the floor can
    never manufacture an overlap; if the gap to the next cue is itself
    smaller than MIN_WORD_DUR, we take what room is available rather than
    forcing overlap.
    """
    tuples = []
    for c in cues:
        start = c["start_frame"] * seconds_per_frame
        end = c["end_frame"] * seconds_per_frame
        tuples.append([c["word"], start, end, c["sura"], c["aya"], c["is_repeat"]])
    tuples.sort(key=lambda t: t[1])

    for i, t in enumerate(tuples):
        start = t[1]
        floored_end = max(t[2], start + MIN_WORD_DUR)
        if i + 1 < len(tuples):
            floored_end = min(floored_end, tuples[i + 1][1])
        t[2] = max(floored_end, start)  # never let end regress below start

    return [tuple(t) for t in tuples]


def emit_json(cues_tuples, out_path):
    data = [
        {"word": w, "start": s, "end": e, "sura": sura, "aya": aya, "is_repeat": rep}
        for (w, s, e, sura, aya, rep) in cues_tuples
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path} ({len(data)} word entries)")
