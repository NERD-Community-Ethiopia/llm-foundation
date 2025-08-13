"""
Text cleaning utilities for Amharic→Oromiffa datasets.

This module provides:
- normalize_text: whitespace and punctuation normalization
- remove_zero_width_chars: strip BOM and zero-width marks
- filter_pair: basic length-ratio and emptiness filtering
"""

from __future__ import annotations

import re
from typing import Tuple


ZERO_WIDTH_PATTERN = re.compile(
    r"[\u200B\u200C\u200D\u200E\u200F\uFEFF]"  # ZWSP, ZWNJ, ZWJ, LRM, RLM, BOM
)


def remove_zero_width_chars(text: str) -> str:
    return ZERO_WIDTH_PATTERN.sub("", text)


def normalize_text(text: str) -> str:
    text = text.strip()
    text = remove_zero_width_chars(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([.,!?;:])\s*", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def filter_pair(src: str, tgt: str, max_len: int = 128, length_ratio: float = 3.0) -> Tuple[bool, str]:
    """
    Basic filtering for parallel pairs.
    - Remove empty sides
    - Remove overly long sentences
    - Remove extreme length ratio mismatches
    Returns (keep, reason_if_dropped)
    """
    if not src or not tgt:
        return False, "empty-side"
    src_len = len(src.split())
    tgt_len = len(tgt.split())
    if src_len == 0 or tgt_len == 0:
        return False, "zero-token"
    if src_len > max_len or tgt_len > max_len:
        return False, "too-long"
    longer = max(src_len, tgt_len)
    shorter = min(src_len, tgt_len)
    if shorter == 0:
        return False, "zero-shorter"
    if (longer / shorter) > length_ratio:
        return False, "length-ratio"
    return True, ""


