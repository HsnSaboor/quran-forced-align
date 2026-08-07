"""Phoneme/letter-tier output builder: groups a word cue's already-computed
per-token frame spans (see repeats.token_frame_spans) by which Uthmani
letter produced them, and attaches the tajweed/silent-letter data
reference.py already carries per letter (see reference.build_text_reference/
build_combined_reference's `letters` field).

This is PURE POST-PROCESSING over data the alignment pipeline already
computed by the time a cue reaches here -- no new Viterbi pass, no new ONNX
inference, no new alignment of any kind. See pipeline.py's DETERMINISM
section and viterbi.py's module docstring for where the actual compute
happens; this module only reshapes already-finished results.

CELL SCHEMA -- one dict per phoneme token, nested under its letter:
    {"phoneme": phoneme_str, "start": start_sec, "end": end_sec}
and one dict per letter, nested under its word (see `build_letter_tier`):
    {"char", "deleted", "start", "end", "tajweed_rules",
     "boundary_tajweed_rules", "phonemes": [cell, ...]}
Deliberately a flat, explicit dict (not a positional tuple like QUA's
9-field cell array) -- this package's letter/phoneme counts per word are
small (a handful at most), so JSON verbosity here is not a real payload-size
concern the way it might be at QUA's reported dataset-publishing scale, and
a dict is self-describing without a separate schema-version document.
"""


def build_letter_tier(cue, combined_token_ids, id2tok, seconds_per_frame):
    """Build the per-letter (and, nested under each letter, per-phoneme)
    tier for one word cue, from data already present on it:
      - cue["letters"]: per-char skeleton (char, deleted, tajweed_rules,
        boundary_tajweed_rules), independent of timing.
      - cue["token_positions"] / cue["token_char_idx"] / cue["token_frame_spans"]:
        parallel arrays, one entry per phoneme TOKEN this word produced,
        giving that token's global position (to look up its phoneme string
        in combined_token_ids/id2tok), which word-local letter it belongs
        to, and its (start_frame, end_frame) span.

    A silently-dropped (deleted) letter never appears in token_char_idx (it
    produced zero phoneme output), so it naturally gets an empty
    `phonemes` list and no start/end -- reported as `None` for both,
    letting a consumer distinguish "this letter is silent" (deleted=True)
    from "this letter has timing" (start/end set) without inferring one
    from the other.

    A diacritic-only letter (haraka/tanween/shadda -- e.g. the fatha in
    "أَ") ALSO gets `start=end=None` despite not being `deleted`: this
    package's phoneme tokens are pre-composed base-letter+diacritic
    clusters (confirmed via tokens.txt, e.g. token "ءَ" = hamza+fatha as
    one CTC symbol), so the diacritic has no separate audio span of its
    own to report -- its timing is inherently the base letter's timing. A
    consumer wanting a single "how long was this letter+its vowel" span
    should read the BASE letter's start/end and treat immediately-following
    diacritic-only entries as zero-duration by construction, not as a gap
    in the alignment.

    Returns a list of letter dicts, one per character of `cue["letters"]`,
    in original (word-text) order.
    """
    letters = cue["letters"]
    token_positions = cue["token_positions"]
    token_char_idx = cue["token_char_idx"]
    token_frame_spans = cue["token_frame_spans"]

    # These three arrays are built in lockstep, one entry per phoneme
    # token, by reference.build_combined_reference (token_positions/
    # token_char_idx) and repeats.token_frame_spans (token_frame_spans) --
    # always the same length for any cue that reaches this point. Asserted
    # explicitly (not just relied upon via zip()'s silent shortest-wins
    # truncation) so a future regression in either producer would fail
    # loudly here instead of silently truncating/misaligning letter-tier
    # output -- consistent with this codebase's existing policy elsewhere
    # (e.g. tokenizer.py raises rather than drop chars) of never
    # swallowing a data-attribution mismatch.
    assert len(token_positions) == len(token_char_idx) == len(token_frame_spans), (
        f"word {cue['word']!r}: token_positions/token_char_idx/token_frame_spans "
        f"length mismatch ({len(token_positions)}/{len(token_char_idx)}/{len(token_frame_spans)}) "
        "-- these must be built in lockstep, one entry per phoneme token"
    )

    phonemes_by_char_idx = {}
    for global_pos, char_idx, (start_frame, end_frame) in zip(
        token_positions, token_char_idx, token_frame_spans,
    ):
        if char_idx is None:
            continue
        phoneme_str = id2tok[combined_token_ids[global_pos]]
        cell = {
            "phoneme": phoneme_str,
            "start": start_frame * seconds_per_frame,
            "end": end_frame * seconds_per_frame,
        }
        phonemes_by_char_idx.setdefault(char_idx, []).append(cell)

    letter_tier = []
    for char_idx, letter in enumerate(letters):
        cells = phonemes_by_char_idx.get(char_idx, [])
        letter_tier.append({
            "char": letter["char"],
            "deleted": letter["deleted"],
            "tajweed_rules": letter["tajweed_rules"],
            "boundary_tajweed_rules": letter["boundary_tajweed_rules"],
            "start": cells[0]["start"] if cells else None,
            "end": cells[-1]["end"] if cells else None,
            "phonemes": cells,
        })
    return letter_tier
