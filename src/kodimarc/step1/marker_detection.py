from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple

from kodimarc.common.markers import ALL_LEX_MARKERS, LEXICON


QUOTE_PREFIX = r"['\"“”‘’\(\[«]*"


def key_to_regex(marker: str) -> str:
    return re.sub(r"\s+", r"\\s+", re.escape(marker))


ALT = "|".join(key_to_regex(marker) for marker in sorted(set(ALL_LEX_MARKERS), key=len, reverse=True))
LEAD_RE = re.compile(rf"^\s*{QUOTE_PREFIX}\s*(?P<m>{ALT})(?P<trail>\s*[,·:;\-–—]?\s*)")
INTRA_RE = re.compile(rf"[,;]\s*(?P<m>{ALT})(?=[,;:)\]»\s])")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_and_strip_marker(hypothesis: str) -> Tuple[Optional[str], str]:
    normalized = normalize_text(hypothesis)

    lead_match = LEAD_RE.match(normalized)
    if lead_match:
        marker = lead_match.group("m")
        stripped = normalized[lead_match.end() :].lstrip(" ,;:·-–—")
        return marker, stripped

    intra_match = INTRA_RE.search(normalized)
    if intra_match:
        marker = intra_match.group("m")
        stripped = normalize_text(normalized[: intra_match.start()] + normalized[intra_match.end() :])
        return marker, stripped

    return None, normalized


def label_of_marker(marker: Optional[str]) -> Optional[str]:
    if marker is None:
        return None
    for label, markers in LEXICON.items():
        if marker in markers:
            return label
    return None
