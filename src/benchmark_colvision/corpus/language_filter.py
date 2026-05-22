"""Language detection on PDF first-page text.

Wraps `langdetect` behind a tolerant API: a single function returns the ISO 639-1
code (e.g. "en") or `None` when detection fails or is below a confidence floor.
"""

from __future__ import annotations

from typing import Optional


# `langdetect`'s detector is stochastic by default; we seed it on first import
# so that the same input always produces the same output.
_SEEDED = False


def _seed_langdetect() -> None:
    global _SEEDED
    if _SEEDED:
        return
    try:
        from langdetect import DetectorFactory  # type: ignore[import-not-found]

        DetectorFactory.seed = 0
    except ImportError:
        pass
    _SEEDED = True


def detect_language(text: str, min_chars: int = 80) -> Optional[str]:
    """Return ISO 639-1 code of `text`, or None if undetectable or too short."""
    if not text or len(text) < min_chars:
        return None
    _seed_langdetect()
    try:
        from langdetect import detect  # type: ignore[import-not-found]
        from langdetect.lang_detect_exception import LangDetectException  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def matches_language(text: str, expected: str, min_chars: int = 80) -> bool:
    """True iff detected language equals `expected` (case-insensitive)."""
    detected = detect_language(text, min_chars=min_chars)
    if detected is None:
        return False
    return detected.lower() == expected.lower()
