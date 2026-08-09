"""Per-ayah word<->phoneme reference builder + whole-surah combined
reference.

  - `text.py`: per-ayah/word Uthmani-text <-> phoneme mapping (fixes a real
    bug found in build_surah_srt.py's build_text_reference() -- see that
    module's docstring for the full bug report and naming-history note).
  - `surah.py`: per-ayah/istiaatha/whole-surah reference assembly.
  - `boundary.py`: cross-ayah-boundary tajweed rule bridging.
  - `combined.py`: the whole-surah flat token-id list + word-slot structure
    `pipeline.align_surah` actually consumes.
"""
from .combined import build_combined_reference
from .surah import build_ayah_reference, build_istiaatha_reference, build_surah_reference
from .text import build_text_reference

__all__ = [
    "build_text_reference",
    "build_ayah_reference",
    "build_istiaatha_reference",
    "build_surah_reference",
    "build_combined_reference",
]
