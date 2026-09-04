"""Learned corrections registry and repeat alignment engine.

Ensures that forced alignment pipelines:
1. Learn from verified corrections so the same mistakes are never repeated on subsequent runs.
2. Maintain strict separation between:
   - `v.words`: EXACTLY 1 entry per canonical word (1-to-1 index matching Uthmani text).
   - `v.segments`: All spoken word instances in chronological order (including repeats).
3. Persist and recall known repeat annotations and timing overrides across reciters and surahs.
"""

import json
import os
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_CORRECTIONS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "learned_corrections.json"
)


def normalize_quran_text(text: str) -> str:
    """Normalize Arabic / Quranic Unicode strings for robust equality matching.
    
    Standardizes Unicode normalization form (NFC), normalizes tanween forms
    (e.g., small meem above vs standard tanween), and strips superfluous spaces.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text.strip())
    t = t.replace("\u06e2", "\u06ed")
    t = t.replace("ثُمَّ", "ثُمَّ")
    return t


def align_spoken_to_canonical(
    canon_words: List[str], spoken_words: List[str]
) -> List[int]:
    """Align spoken words to canonical words using dynamic programming with repeat backtrack.
    
    Returns a list of 1-based canonical word indices `wi` for each spoken word.
    """
    n = len(canon_words)
    m = len(spoken_words)
    if m == 0 or n == 0:
        return []

    norm_canon = [normalize_quran_text(w) for w in canon_words]
    norm_spoken = [normalize_quran_text(w) for w in spoken_words]

    dp = [[float("inf")] * (n + 1) for _ in range(m + 1)]
    parent = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0

    for i in range(1, m + 1):
        spk = norm_spoken[i - 1]
        for j in range(1, n + 1):
            can = norm_canon[j - 1]
            match_cost = 0 if spk == can else 1000

            cost_fwd = dp[i - 1][j - 1] + match_cost
            best_prev = j - 1
            best_cost = cost_fwd

            for k in range(j, n + 1):
                cost_rep = dp[i - 1][k] + match_cost + 5
                if cost_rep < best_cost:
                    best_cost = cost_rep
                    best_prev = k

            dp[i][j] = best_cost
            parent[i][j] = best_prev

    best_j = min(range(1, n + 1), key=lambda j: dp[m][j])
    path = []
    curr_j = best_j
    for i in range(m, 0, -1):
        path.append(curr_j)
        curr_j = parent[i][curr_j]
    path.reverse()
    return path


class LearnedCorrectionsRegistry:
    """Persistent storage and lookup for learned recitation corrections and repeat sites."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else _DEFAULT_CORRECTIONS_PATH
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as e:
                print(f"[LearnedCorrectionsRegistry] Warning: could not load {self.path}: {e}")
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get_corrections(
        self, reciter: str, surah_id: int, ayah_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        rec_data = self._data.get(reciter, {}).get(str(surah_id), {})
        if ayah_id is not None:
            return rec_data.get(str(ayah_id), [])
        return rec_data

    def register_repeat(
        self,
        reciter: str,
        surah_id: int,
        ayah_id: int,
        word_indices: List[int],
        phrase: str,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
    ) -> None:
        """Register a known repeated phrase in an ayah."""
        rec_data = self._data.setdefault(reciter, {}).setdefault(str(surah_id), {}).setdefault(str(ayah_id), [])
        for entry in rec_data:
            if entry.get("words") == word_indices and entry.get("type") == "repeat":
                if start_ms is not None:
                    entry["start_ms"] = start_ms
                if end_ms is not None:
                    entry["end_ms"] = end_ms
                self.save()
                return

        rec_data.append({
            "type": "repeat",
            "words": word_indices,
            "phrase": phrase,
            "start_ms": start_ms,
            "end_ms": end_ms,
        })
        self.save()


default_registry = LearnedCorrectionsRegistry()
