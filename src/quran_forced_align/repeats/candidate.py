from ..trellis import frame_spans_from_path
from .spans import token_frame_spans


def build_phrase_ids(word_indices, cues, combined_token_ids):
    """Build the single-copy token-id sequence for the candidate phrase at
    `word_indices` (the concatenation, in word order, of each word's
    tokens' ids) and its doubled copy. Extracted from
    `_repeat_window_candidate` so detect_and_fix_repeats' K-search can run
    the free-decode cross-check (FIX 5) BEFORE paying for the forced
    alignment -- the gate needs these two id lists, which the candidate
    function itself used to build at the top of every call. Returns
    (phrase_token_ids, doubled_ids); both are plain Python lists in the
    exact order the old in-candidate build produced."""
    phrase_token_ids = []
    for j in word_indices:
        phrase_token_ids.extend(combined_token_ids[p] for p in cues[j]["token_positions"])
    doubled_ids = phrase_token_ids + phrase_token_ids
    return phrase_token_ids, doubled_ids


def _repeat_window_candidate(engine, word_indices, cues, log_probs, combined_token_ids, blank_id,
                              window_start, window_end, min_word_dur_frames,
                              phrase_token_ids=None, doubled_ids=None):
    """Try ONE candidate repeated-phrase hypothesis: the contiguous words at
    `word_indices` (indices into `cues`, in ascending order -- may be a
    single word or a multi-word phrase) doubled back-to-back and re-aligned
    against the frame span [window_start, window_end] of `log_probs`.

    `engine` is whichever `engines.Engine` (CPU or CUDA) `align_surah` was
    called with -- this local re-alignment must use the SAME engine as the
    main pass, since `engine.forced_align`'s `margins` semantics differ
    between engines (see `engines/cuda.py`'s module docstring) and mixing
    them within one surah's repeat-detection pass would silently compare
    margins from two different confidence scales.

    `phrase_token_ids`/`doubled_ids`, if given, are the candidate phrase's
    id lists (from `build_phrase_ids`) -- the K-search caller computes them
    already for its free-decode gate and passes them in so they aren't
    rebuilt here; when absent they are built here, exactly as before.

    This is the shared core the K=1 (single-word) doubling used to do
    inline; it's now extracted so the K-search loop in
    detect_and_fix_repeats can call it once per candidate window size
    without duplicating the trellis-construction/state-bookkeeping logic.

    Returns None if the doubled alignment fails outright, any word's tokens
    never got a state in the local trellis, or the two copies aren't
    timing-plausible (non-overlapping, each occupying >= min_word_dur_frames
    overall -- deliberately NOT scaled up by the number of words: this is
    the same single-word floor the original K=1 check used, applied to the
    whole phrase span, so it stays at least as permissive for K>1 as it was
    for K=1). Does NOT apply the acoustic-confidence gate -- that is the
    caller's job, so every K can be compared against the SAME floor on an
    equal footing.

    On success, returns a dict with the window's local ext/path/log_probs
    (for the caller to compute the confidence-gate averages), the whole-copy
    local spans (for the confidence gate), and a per-word breakdown of both
    copies' local (start, end) frame spans (for splicing individual word
    cues back in).
    """
    word_ntoks = [len(cues[j]["token_positions"]) for j in word_indices]
    if any(nt == 0 for nt in word_ntoks):
        return None
    offsets = []
    acc = 0
    for nt in word_ntoks:
        offsets.append(acc)
        acc += nt
    L = acc

    if phrase_token_ids is None or doubled_ids is None:
        phrase_token_ids, doubled_ids = build_phrase_ids(word_indices, cues, combined_token_ids)

    window_log_probs = log_probs[window_start:window_end + 1]
    ext2, path2, margins2 = engine.forced_align(window_log_probs, doubled_ids, blank_id)
    if ext2 is None:
        return None

    num_states = len(ext2)
    first_seen, last_seen = frame_spans_from_path(path2, num_states)

    def positions_for(local_offset, count):
        return list(range(local_offset, local_offset + count))

    copy1_positions = positions_for(0, L)
    copy2_positions = positions_for(L, L)
    copy1_spans = token_frame_spans(copy1_positions, first_seen, last_seen)
    copy2_spans = token_frame_spans(copy2_positions, first_seen, last_seen)
    if copy1_spans is None or copy2_spans is None:
        return None
    _copy1_token_spans, copy1_start_local, copy1_end_local = copy1_spans
    _copy2_token_spans, copy2_start_local, copy2_end_local = copy2_spans

    timing_plausible = (
        copy2_start_local > copy1_end_local
        and (copy1_end_local - copy1_start_local) >= min_word_dur_frames
        and (copy2_end_local - copy2_start_local) >= min_word_dur_frames
    )
    if not timing_plausible:
        return None

    per_word_copy1 = {}
    per_word_copy2 = {}
    for m, j in enumerate(word_indices):
        nt = word_ntoks[m]
        s1_spans = token_frame_spans(positions_for(offsets[m], nt), first_seen, last_seen)
        s2_spans = token_frame_spans(positions_for(L + offsets[m], nt), first_seen, last_seen)
        s1_token_spans, s1_start, s1_end = s1_spans
        s2_token_spans, s2_start, s2_end = s2_spans
        per_word_copy1[j] = (s1_start, s1_end, s1_token_spans)
        per_word_copy2[j] = (s2_start, s2_end, s2_token_spans)

    return {
        "window_log_probs": window_log_probs,
        "ext": ext2,
        "path": path2,
        "margins": margins2,
        "copy1_start_local": copy1_start_local,
        "copy1_end_local": copy1_end_local,
        "copy2_start_local": copy2_start_local,
        "copy2_end_local": copy2_end_local,
        "per_word_copy1": per_word_copy1,
        "per_word_copy2": per_word_copy2,
        # The candidate phrase's id lists -- built here (or passed in from
        # the caller's free-decode gate via `build_phrase_ids`) and exposed
        # so the gate's thresholds stay checkable against the exact ids
        # this alignment actually used.
        "phrase_token_ids": phrase_token_ids,
        "doubled_ids": doubled_ids,
    }
