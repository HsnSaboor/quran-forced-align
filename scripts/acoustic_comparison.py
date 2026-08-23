#!/usr/bin/env python3
"""Acoustic and Alignment Quality Comparison Analysis for Surah 2 Alignment Outputs.
"""
import json
import numpy as np

def analyze_reciter(json_path, name):
    with open(json_path, "r", encoding="utf-8") as f:
        cues = json.load(f)
        
    total_cues = len(cues)
    repeats = [c for c in cues if c.get("is_repeat")]
    non_repeats = [c for c in cues if not c.get("is_repeat")]
    low_conf = [c for c in cues if c.get("low_confidence")]
    
    # Word Durations
    durations = [c["end"] - c["start"] for c in cues if c.get("start") is not None and c.get("end") is not None]
    durations_np = np.array(durations)
    
    # Acoustic Log-Probs
    logprobs = [c["avg_logprob"] for c in cues if c.get("avg_logprob") is not None]
    logprobs_np = np.array(logprobs)
    
    # Confidence tiers
    high_conf = [lp for lp in logprobs if lp >= -3.0]
    med_conf = [lp for lp in logprobs if -5.0 <= lp < -3.0]
    low_conf_tier = [lp for lp in logprobs if lp < -5.0]
    
    # Pauses between consecutive words
    pauses = []
    monotonic_errors = 0
    overlap_errors = 0
    for i in range(len(cues) - 1):
        c1 = cues[i]
        c2 = cues[i+1]
        if c1.get("end") is not None and c2.get("start") is not None:
            gap = c2["start"] - c1["end"]
            if gap < -0.001:
                overlap_errors += 1
            if c2["start"] < c1["start"]:
                monotonic_errors += 1
            if gap >= 0:
                pauses.append(gap)
                
    pauses_np = np.array(pauses) if pauses else np.array([0.0])
    
    # Audio span
    first_start = cues[0]["start"] if cues else 0.0
    last_end = cues[-1]["end"] if cues else 0.0
    total_span_sec = last_end - first_start
    total_speaking_sec = np.sum(durations_np)
    total_pause_sec = np.sum(pauses_np)
    
    # Repeat clusters & phrase lengths
    repeat_phrase_lengths = {}
    cur_streak = 0
    for c in cues:
        if c.get("is_repeat"):
            cur_streak += 1
        else:
            if cur_streak > 0:
                repeat_phrase_lengths[cur_streak] = repeat_phrase_lengths.get(cur_streak, 0) + 1
                cur_streak = 0
    if cur_streak > 0:
        repeat_phrase_lengths[cur_streak] = repeat_phrase_lengths.get(cur_streak, 0) + 1
        
    return {
        "name": name,
        "total_cues": total_cues,
        "base_words": len(non_repeats),
        "repeat_words": len(repeats),
        "low_conf_words": len(low_conf),
        "low_conf_pct": 100 * len(low_conf) / max(1, total_cues),
        "total_span_sec": total_span_sec,
        "speaking_sec": total_speaking_sec,
        "pause_sec": total_pause_sec,
        "speaking_ratio": total_speaking_sec / max(1e-6, total_span_sec),
        "wpm": (total_cues / (total_span_sec / 60.0)) if total_span_sec > 0 else 0,
        "dur_mean": float(np.mean(durations_np)),
        "dur_median": float(np.median(durations_np)),
        "dur_p5": float(np.percentile(durations_np, 5)),
        "dur_p95": float(np.percentile(durations_np, 95)),
        "dur_min": float(np.min(durations_np)),
        "dur_max": float(np.max(durations_np)),
        "logp_mean": float(np.mean(logprobs_np)),
        "logp_median": float(np.median(logprobs_np)),
        "logp_std": float(np.std(logprobs_np)),
        "high_conf_pct": 100 * len(high_conf) / max(1, len(logprobs)),
        "med_conf_pct": 100 * len(med_conf) / max(1, len(logprobs)),
        "low_conf_pct_tier": 100 * len(low_conf_tier) / max(1, len(logprobs)),
        "pause_mean": float(np.mean(pauses_np)),
        "pause_median": float(np.median(pauses_np)),
        "pause_max": float(np.max(pauses_np)),
        "monotonic_errors": monotonic_errors,
        "overlap_errors": overlap_errors,
        "repeat_phrase_lengths": repeat_phrase_lengths,
    }

def main():
    reciters = [
        ("verification_output/abdallah-kamel_002_aligned.json", "Abdallah Kamel"),
        ("verification_output/abdel-mohsen-al-obeikan_002_aligned.json", "Abdel-Mohsen Al-Obeikan"),
    ]
    
    stats = [analyze_reciter(path, name) for path, name in reciters]
    
    print("=" * 90)
    print("SURAH 2 AL-BAQARAH: ACOUSTIC & ALIGNMENT QUALITY COMPARISON")
    print("=" * 90)
    
    header = f"{'Metric':<35} | {'Abdallah Kamel':<24} | {'Abdel-Mohsen Al-Obeikan':<24}"
    print(header)
    print("-" * 90)
    
    rows = [
        ("Total Audio Span", f"{stats[0]['total_span_sec']:.1f}s ({stats[0]['total_span_sec']/60:.1f} min)", f"{stats[1]['total_span_sec']:.1f}s ({stats[1]['total_span_sec']/60:.1f} min)"),
        ("Net Speaking Time", f"{stats[0]['speaking_sec']:.1f}s ({100*stats[0]['speaking_ratio']:.1f}%)", f"{stats[1]['speaking_sec']:.1f}s ({100*stats[1]['speaking_ratio']:.1f}%)"),
        ("Total Pause / Silence", f"{stats[0]['pause_sec']:.1f}s", f"{stats[1]['pause_sec']:.1f}s"),
        ("Recitation Tempo (WPM)", f"{stats[0]['wpm']:.1f} words/min", f"{stats[1]['wpm']:.1f} words/min"),
        ("Total Aligned Cues", f"{stats[0]['total_cues']}", f"{stats[1]['total_cues']}"),
        ("Base Quran Words", f"{stats[0]['base_words']}", f"{stats[1]['base_words']}"),
        ("Repeated Words Spliced", f"{stats[0]['repeat_words']} ({100*stats[0]['repeat_words']/stats[0]['total_cues']:.2f}%)", f"{stats[1]['repeat_words']} ({100*stats[1]['repeat_words']/stats[1]['total_cues']:.2f}%)"),
        ("Repeat Phrases Count", f"{sum(stats[0]['repeat_phrase_lengths'].values())} phrases", f"{sum(stats[1]['repeat_phrase_lengths'].values())} phrases"),
        ("Max Repeat Phrase Length", f"{max(stats[0]['repeat_phrase_lengths'].keys(), default=0)} words", f"{max(stats[1]['repeat_phrase_lengths'].keys(), default=0)} words"),
        ("Mean Word Duration", f"{stats[0]['dur_mean']:.3f}s", f"{stats[1]['dur_mean']:.3f}s"),
        ("Median Word Duration", f"{stats[0]['dur_median']:.3f}s", f"{stats[1]['dur_median']:.3f}s"),
        ("Word Duration (p5 - p95)", f"{stats[0]['dur_p5']:.2f}s - {stats[0]['dur_p95']:.2f}s", f"{stats[1]['dur_p5']:.2f}s - {stats[1]['dur_p95']:.2f}s"),
        ("Mean Acoustic Log-Prob", f"{stats[0]['logp_mean']:.3f}", f"{stats[1]['logp_mean']:.3f}"),
        ("Median Acoustic Log-Prob", f"{stats[0]['logp_median']:.3f}", f"{stats[1]['logp_median']:.3f}"),
        ("High Conf Words (logp >= -3)", f"{stats[0]['high_conf_pct']:.1f}%", f"{stats[1]['high_conf_pct']:.1f}%"),
        ("Medium Conf Words ([-5, -3))", f"{stats[0]['med_conf_pct']:.1f}%", f"{stats[1]['med_conf_pct']:.1f}%"),
        ("Low Conf Flagged (logp < -5)", f"{stats[0]['low_conf_pct_tier']:.1f}%", f"{stats[1]['low_conf_pct_tier']:.1f}%"),
        ("Overlap / Monotonic Errors", f"{stats[0]['overlap_errors']} / {stats[0]['monotonic_errors']}", f"{stats[1]['overlap_errors']} / {stats[1]['monotonic_errors']}"),
    ]
    
    for r in rows:
        print(f"{r[0]:<35} | {r[1]:<24} | {r[2]:<24}")
    print("=" * 90)
    
    print("\nRepeat Phrase Length Distribution:")
    for s in stats:
        print(f"\n  {s['name']}:")
        for k in sorted(s['repeat_phrase_lengths'].keys()):
            count = s['repeat_phrase_lengths'][k]
            print(f"    Length {k:>2} words: {count:>3} occurrences ({count*k:>3} total repeated words)")

if __name__ == "__main__":
    main()
