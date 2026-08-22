"""tokens.txt -> char-sequence tokenizer (longest-match / max-munge)."""
from functools import lru_cache


@lru_cache(maxsize=16)
def load_tokens(tokens_path):
    """Parse tokens.txt into (tok2id, id2tok, blank_id, max_token_len).

    tokens.txt lines are "<token-string> <id>". Most tokens are multi-char
    (already-merged phoneme clusters, e.g. shadda-doubled letters, madd
    elongations); a handful are single chars. <blank> is the CTC blank
    symbol. We need the id->token map for decoding, the token->id map plus
    the max token length for a greedy longest-match tokenizer that turns a
    plain phoneme-char string (as produced by reference.build_text_reference,
    which strips spaces) into token ids identical to the model's own output
    vocabulary.
    """
    tok2id = {}
    id2tok = {}
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            tok, idx_s = line.rsplit(" ", 1)
            idx = int(idx_s)
            tok2id[tok] = idx
            id2tok[idx] = tok
    blank_id = tok2id["<blank>"]
    max_token_len = max(len(t) for t in tok2id if t != "<blank>")
    return tok2id, id2tok, blank_id, max_token_len


def tokenize_with_char_starts(phonemes, tok2id, max_token_len):
    """Greedy longest-match tokenize a phoneme-char string into token ids,
    also returning the starting char index of each token (needed to look up
    which reference word each token belongs to via phoneme_to_word, which
    is indexed per-char by build_text_reference).

    Every char must match some token exactly at that position -- there is no
    silent fallback per-char token in tokens.txt (single Arabic letters
    without diacritics are NOT valid standalone tokens for most letters;
    only the pre-composed phoneme clusters are). If a position can't be
    matched at all, this is a genuine data problem (a phoneme sequence that
    the model's vocabulary can't express) and we raise loudly rather than
    silently dropping chars, since that would corrupt alignment.
    """
    ids = []
    char_starts = []
    i = 0
    n = len(phonemes)
    while i < n:
        matched = False
        for length in range(min(max_token_len, n - i), 0, -1):
            cand = phonemes[i:i + length]
            if cand in tok2id:
                ids.append(tok2id[cand])
                char_starts.append(i)
                i += length
                matched = True
                break
        if not matched:
            raise ValueError(
                f"no token in tokens.txt matches phoneme char {phonemes[i]!r} "
                f"(U+{ord(phonemes[i]):04X}) at position {i} in {phonemes!r}"
            )
    return ids, char_starts
