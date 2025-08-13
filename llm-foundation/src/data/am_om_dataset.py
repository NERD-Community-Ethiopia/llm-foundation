"""
Amharic→Oromiffa dataset utilities.

This module provides minimal data loading, normalization, tokenization,
vocabulary building, and conversion to model-ready numpy arrays.

It intentionally starts with a whitespace-based tokenizer to keep the
pipeline simple and dependency-light. You can later swap in SentencePiece
by replacing the tokenizer implementation while preserving the API.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional

import numpy as np


# Special tokens and their canonical ids used across the project
PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"


def normalize_text(text: str) -> str:
    """
    Basic text normalization suitable for both Amharic and Oromiffa.
    - Trim whitespace
    - Collapse multiple spaces
    - Normalize common punctuation spacing
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Normalize spaces around punctuation
    text = re.sub(r"\s*([.,!?;:])\s*", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_tsv(files: Iterable[str], delimiter: str = "\t") -> List[Tuple[str, str]]:
    """
    Load parallel sentences from tab-separated files.

    Each line: <amharic><TAB><oromiffa>
    Returns list of (amharic, oromiffa) tuples.
    """
    pairs: List[Tuple[str, str]] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split(delimiter)
                if len(parts) < 2:
                    continue
                src, tgt = parts[0], parts[1]
                pairs.append((normalize_text(src), normalize_text(tgt)))
    return pairs


class WhitespaceTokenizer:
    """
    Simple whitespace tokenizer with a learned vocabulary from training data.
    The first four ids are reserved for special tokens in this order:
    <PAD>=0, <START>=1, <END>=2, <UNK>=3
    """

    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    @staticmethod
    def build_from_corpus(corpus_texts: Iterable[str], max_vocab_size: int = 32000) -> "WhitespaceTokenizer":
        frequencies: Dict[str, int] = {}
        for text in corpus_texts:
            for token in text.split():
                frequencies[token] = frequencies.get(token, 0) + 1

        # Reserve special tokens first
        token_to_id: Dict[str, int] = {
            PAD_TOKEN: 0,
            START_TOKEN: 1,
            END_TOKEN: 2,
            UNK_TOKEN: 3,
        }

        # Sort tokens by frequency then alphabetically for determinism
        sorted_tokens = sorted(frequencies.items(), key=lambda kv: (-kv[1], kv[0]))
        for token, _ in sorted_tokens:
            if token in token_to_id:
                continue
            if len(token_to_id) >= max_vocab_size:
                break
            token_to_id[token] = len(token_to_id)

        return WhitespaceTokenizer(token_to_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = [self.token_to_id.get(tok, self.token_to_id[UNK_TOKEN]) for tok in text.split()]
        if add_special_tokens:
            ids = [self.token_to_id[START_TOKEN]] + ids + [self.token_to_id[END_TOKEN]]
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens: List[str] = []
        for i in ids:
            token = self.id_to_token.get(int(i), UNK_TOKEN)
            if skip_special_tokens and token in {PAD_TOKEN, START_TOKEN, END_TOKEN}:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> "WhitespaceTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            token_to_id = json.load(f)
        return WhitespaceTokenizer(token_to_id)


@dataclass
class EncodedSample:
    source_ids: np.ndarray  # shape (src_len,)
    target_ids: np.ndarray  # shape (tgt_len,)


def pad_sequence(ids: List[int], desired_length: int, pad_id: int) -> np.ndarray:
    if len(ids) >= desired_length:
        return np.asarray(ids[:desired_length], dtype=np.int32)
    padded = ids + [pad_id] * (desired_length - len(ids))
    return np.asarray(padded, dtype=np.int32)


def encode_pairs(
    pairs: List[Tuple[str, str]],
    tokenizer: WhitespaceTokenizer,
    max_src_len: int,
    max_tgt_len: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Convert text pairs to (src_ids, tgt_ids), padding to fixed lengths.
    Target ids include START/END tokens; source does not.
    """
    pad_id = tokenizer.token_to_id[PAD_TOKEN]
    encoded: List[Tuple[np.ndarray, np.ndarray]] = []
    for src_text, tgt_text in pairs:
        src_ids = tokenizer.encode(src_text, add_special_tokens=False)
        tgt_ids = tokenizer.encode(tgt_text, add_special_tokens=True)
        src_arr = pad_sequence(src_ids, max_src_len, pad_id)
        tgt_arr = pad_sequence(tgt_ids, max_tgt_len, pad_id)
        encoded.append((src_arr, tgt_arr))
    return encoded


def build_tokenizer_from_pairs(
    train_pairs: List[Tuple[str, str]],
    max_vocab_size: int = 32000,
) -> WhitespaceTokenizer:
    corpus_iter = (normalize_text(s) for pair in train_pairs for s in pair)
    return WhitespaceTokenizer.build_from_corpus(corpus_iter, max_vocab_size=max_vocab_size)


