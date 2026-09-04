import numpy as np


def token_frame_spans(token_positions, first_seen, last_seen):
    """For a list of token positions (indices into a combined token-id
    list), return (per_token_spans, word_start, word_end) where
    per_token_spans[i] = (start_frame, end_frame) for token_positions[i]
    and word_start/word_end are the min/max over all of them. Token
    position p (an index into the combined token-id list) lives at
    extended-trellis label-state 2*p+1.

    Returns None if any token never got a state in the trellis (start<0),
    the same failure signal both `extract_word_frame_spans` and
    `_repeat_window_candidate` already relied on before this was factored
    out -- shared here so the phoneme-tier span data (per_token_spans) is
    computed identically, once, wherever a word's frame span is derived
    from token-level trellis states, instead of two near-duplicate min/max
    loops that could drift apart under future edits.

    Vectorized with numpy fancy indexing over the label states 2*p+1 --
    one gather per of first_seen/last_seen instead of a Python loop over
    the token positions. The return contract is unchanged: a list of
    (int, int) tuples, or None.
    """
    states = 2 * np.asarray(token_positions, dtype=np.int64) + 1
    starts = first_seen[states]
    ends = last_seen[states]
    if starts.size == 0 or np.any(starts < 0) or np.any(ends < 0):
        return None
    per_token_spans = list(zip(starts.tolist(), ends.tolist()))
    word_start = int(np.min(starts))
    word_end = int(np.max(ends))
    return per_token_spans, word_start, word_end


def extract_word_frame_spans(word_slots, first_seen, last_seen):
    """Vectorized high-performance extraction of per-word frame spans from first_seen and last_seen arrays."""
    cues = []
    for slot in word_slots:
        positions = slot.get("token_positions")
        if not positions:
            continue
        
        p0 = positions[0]
        p_last = positions[-1]
        s0 = 2 * p0 + 1
        s_last = 2 * p_last + 1
        
        w_start = int(first_seen[s0])
        w_end = int(last_seen[s_last])
        if w_start < 0 or w_end < 0:
            continue
            
        per_token_spans = [
            (int(first_seen[2 * p + 1]), int(last_seen[2 * p + 1]))
            for p in positions
        ]
        
        cues.append({
            "word": slot["word"],
            "sura": slot["sura"],
            "aya": slot["aya"],
            "word_idx": slot.get("word_idx"),
            "is_ayah_final": slot["is_ayah_final"],
            "start_frame": w_start,
            "end_frame": w_end,
            "is_repeat": False,
            "token_positions": positions,
            "token_char_idx": slot["token_char_idx"],
            "letters": slot["letters"],
            "token_frame_spans": per_token_spans,
        })
    return cues
