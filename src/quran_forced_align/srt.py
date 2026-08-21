"""SRT/JSON emission: timestamp formatting, SRT writer, frame-cue -> tuple
conversion, JSON writer.

`fmt_srt_time` and `emit_srt` re-implement the small, generic SRT-writer
utilities that lived in the older build_surah_srt.py script (that script's
own alignment logic is dead weight and intentionally left behind -- only
this generic formatting/writing logic is reused here).
"""
import json
import math

from .cells import build_letter_tier
from .constants import MIN_WORD_DUR


def fmt_srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    mi, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{mi:02d}:{s:02d},{ms:03d}"


def emit_srt(records, out_path):
    """Write plain-text SRT captions from `build_rich_records`'s output.
    SRT format itself has no room for the confidence/letter-tier fields,
    so this only ever reads word/start/end/is_repeat."""
    lines = []
    for idx, r in enumerate(records, 1):
        tag = " [repeat]" if r["is_repeat"] else ""
        lines.append(f"{idx}\n{fmt_srt_time(r['start'])} --> {fmt_srt_time(r['end'])}\n{r['word']}{tag}\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path} ({len(records)} word cues, "
          f"{sum(1 for r in records if r['is_repeat'])} flagged as repeats)")


def _sorted_clamped_spans(cues, seconds_per_frame):
    """Shared core of both output paths below: sort cues by start time and
    compute each one's (start, end) in seconds, with the MIN_WORD_DUR floor
    +neighbour-clamp applied to `end`.

    WHY THE CLAMP EXISTS: raw frame-derived spans from the Viterbi path are
    monotonic and non-overlapping by construction (each frame is assigned
    to exactly one trellis state, and states map 1:1 to word-owned token
    positions in strictly increasing order). The MIN_WORD_DUR floor (padding
    degenerate near-zero-length cues, e.g. a word compressed into a single
    frame) can push a word's `end` PAST the next word's `start` if applied
    naively -- confirmed empirically on the Al-Fatiha smoke test (6/34 cues
    overlapped before this clamp was added). Fix: after sorting, clamp each
    cue's floored end to the next cue's start so the floor can never
    manufacture an overlap; if the gap to the next cue is itself smaller
    than MIN_WORD_DUR, we take what room is available rather than forcing
    overlap.

    Returns a list of (cue, start, end) triples in sorted order; `cue` is
    the ORIGINAL dict (untouched) so callers that need its other fields
    (letters, avg_logprob, ...) still have them.
    """
    rows = [(c, c["start_frame"] * seconds_per_frame, c["end_frame"] * seconds_per_frame) for c in cues]
    rows.sort(key=lambda r: r[1])

    clamped = []
    for i, (c, start, end) in enumerate(rows):
        floored_end = max(end, start + MIN_WORD_DUR)
        if i + 1 < len(rows):
            floored_end = min(floored_end, rows[i + 1][1])
        clamped.append((c, start, max(floored_end, start)))
    return clamped


def cues_to_tuples(cues, seconds_per_frame):
    """Convert frame-indexed cues to (word, start, end, sura, aya, is_repeat)
    tuples, sorted by start time -- see `_sorted_clamped_spans` for the
    sort/clamp logic this delegates to. Used by the manual (non-pipeline)
    ground-truth test harness; the CLI/batch pipeline uses
    `build_rich_records` instead."""
    return [
        (c["word"], start, end, c["sura"], c["aya"], c["is_repeat"])
        for c, start, end in _sorted_clamped_spans(cues, seconds_per_frame)
    ]


def build_rich_records(cues, seconds_per_frame, combined_token_ids, id2tok):
    """Build the full per-word output records (word/timing/repeat flag,
    confidence signals, and the letter/phoneme/tajweed tier), sorted and
    end-clamped identically to `cues_to_tuples` -- the flat fields the web
    player already reads (`word`/`start`/`end`/`sura`/`aya`/`is_repeat`)
    are unchanged; this only ADDS fields, never renames or removes the
    ones downstream code already depends on.

    `combined_token_ids`/`id2tok` are needed to resolve each phoneme
    token's global position back to its actual phoneme string -- see
    `cells.build_letter_tier`.
    """
    records = []
    for c, start, end in _sorted_clamped_spans(cues, seconds_per_frame):
        records.append({
            "word": c["word"],
            "start": start,
            "end": end,
            "sura": c["sura"],
            "aya": c["aya"],
            "is_repeat": c["is_repeat"],
            "avg_logprob": c["avg_logprob"],
            "min_decision_margin": c["min_decision_margin"],
            "low_confidence": c["low_confidence"],
            "letters": build_letter_tier(c, combined_token_ids, id2tok, seconds_per_frame),
        })
    return records


def _json_safe_float(x):
    """Python's `json` module happily serializes float('inf')/float('-inf')
    as the bare (non-standard-JSON) tokens `Infinity`/`-Infinity` -- valid
    for `json.loads` to read back, but REJECTED by strict parsers most
    other consumers use, notably `JSON.parse` in any browser/Node (the web
    player's actual consumer). `avg_logprob` (-inf for a degenerate empty
    span) and `min_decision_margin` (+inf when a word's span includes a
    frame with only one finite backtrace candidate -- always true for
    frame 0, see viterbi.py) can legitimately be infinite, so this
    substitutes `None` (renders as JSON `null`) for either sign of
    infinity before writing, rather than emitting non-standard JSON."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def emit_json_rich(records, out_path):
    """Write `build_rich_records`'s output as strict, standards-compliant
    JSON (see `_json_safe_float` for why a plain `json.dump` isn't enough).
    Uses high-speed SIMD `orjson` when available for instant serialization."""
    safe_records = [
        {**r, "avg_logprob": _json_safe_float(r["avg_logprob"]),
         "min_decision_margin": _json_safe_float(r["min_decision_margin"])}
        for r in records
    ]
    try:
        import orjson
        with open(out_path, "wb") as f:
            f.write(orjson.dumps(safe_records, option=orjson.OPT_INDENT_2))
    except Exception:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(safe_records, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path} ({len(records)} word entries, rich)")
