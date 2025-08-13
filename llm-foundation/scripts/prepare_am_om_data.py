#!/usr/bin/env python3
"""
Data preparation script for Amharic→Oromiffa.

Steps:
  1) Read raw TSV(s) with columns: <amharic>\t<oromiffa>
  2) Clean and filter pairs
  3) Split into train/val/test
  4) Train SentencePiece on train text (optional)
  5) Output cleaned TSVs and SentencePiece model

Usage example:
  python scripts/prepare_am_om_data.py \
    --input raw/am_om.tsv \
    --out-dir data/processed/am-om \
    --spm --vocab-size 32000
"""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Tuple

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data.text_cleaning import normalize_text, filter_pair
from data.sentencepiece_prep import write_corpus_for_spm, train_sentencepiece


def read_tsv(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            pairs.append((parts[0], parts[1]))
    return pairs


def write_tsv(pairs: List[Tuple[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a, b in pairs:
            f.write(f"{a}\t{b}\n")


def split_pairs(pairs: List[Tuple[str, str]], val_ratio: float, test_ratio: float, seed: int = 42):
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = pairs[:n_test]
    val = pairs[n_test:n_test + n_val]
    train = pairs[n_test + n_val:]
    return train, val, test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, nargs="+", help="Raw TSV files (am\tom)")
    p.add_argument("--out-dir", required=True, help="Output directory for cleaned/split data")
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--length-ratio", type=float, default=3.0)
    p.add_argument("--spm", action="store_true", help="Train SentencePiece on train set")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--model-type", default="unigram", choices=["unigram", "bpe", "char", "word"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load
    raw_pairs: List[Tuple[str, str]] = []
    for p in args.input:
        raw_pairs.extend(read_tsv(p))

    # Clean & filter
    cleaned: List[Tuple[str, str]] = []
    dropped = 0
    for am, om in raw_pairs:
        am_n = normalize_text(am)
        om_n = normalize_text(om)
        keep, _ = filter_pair(am_n, om_n, max_len=args.max_len, length_ratio=args.length_ratio)
        if keep:
            cleaned.append((am_n, om_n))
        else:
            dropped += 1

    # Split
    train, val, test = split_pairs(cleaned, args.val_ratio, args.test_ratio)

    # Write splits
    write_tsv(train, os.path.join(args.out_dir, "train.tsv"))
    write_tsv(val, os.path.join(args.out_dir, "val.tsv"))
    write_tsv(test, os.path.join(args.out_dir, "test.tsv"))

    print(f"✅ Cleaned: {len(cleaned)} pairs (dropped: {dropped})")
    print(f"→ Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # SentencePiece
    if args.spm:
        corpus_path = os.path.join(args.out_dir, "train.txt")
        write_corpus_for_spm((s for pair in train for s in pair), corpus_path)
        model_prefix = os.path.join(args.out_dir, "am_om_sp")
        train_sentencepiece(
            input_path=corpus_path,
            model_prefix=model_prefix,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=1.0,
            user_defined_symbols=[],
        )
        print(f"✅ SentencePiece model saved: {model_prefix}.model")


if __name__ == "__main__":
    main()


