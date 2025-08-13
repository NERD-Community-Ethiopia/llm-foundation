#!/usr/bin/env python3
"""
Train a Transformer on Amharic→Oromiffa using the existing numpy-based model.

Usage:
  python scripts/train_am_om.py --train data/am_om_train.tsv --val data/am_om_val.tsv \
      --config configs/transformer_am_om.json --tokenizer-output data/am_om_ws_tokenizer.json

Note: This uses a whitespace tokenizer as a starting point. You can later
replace it with SentencePiece without changing the Trainer/Transformer APIs.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from transformer.core.transformer import Transformer
from transformer.training.training_pipeline import TransformerTrainer
from data.am_om_dataset import (
    load_tsv,
    build_tokenizer_from_pairs,
    WhitespaceTokenizer,
    encode_pairs,
    PAD_TOKEN,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, nargs="+", help="Train TSV files (am\tom)")
    p.add_argument("--val", required=False, nargs="+", help="Validation TSV files (am\tom)")
    p.add_argument("--config", required=True, help="JSON config path")
    p.add_argument("--tokenizer-output", required=True, help="Path to save whitespace tokenizer")
    p.add_argument("--output-dir", default="checkpoints/am-om", help="Where to save checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Load data
    train_pairs = load_tsv(args.train)
    val_pairs: List[Tuple[str, str]] = []
    if args.val:
        val_pairs = load_tsv(args.val)

    # Build tokenizer on training data
    tokenizer: WhitespaceTokenizer = build_tokenizer_from_pairs(train_pairs, max_vocab_size=cfg.get("vocab_size", 32000))
    tokenizer.save(args.tokenizer_output)

    # Prepare encoded/padded arrays
    max_src_len = cfg.get("max_seq_length", 128)
    max_tgt_len = cfg.get("max_seq_length", 128)
    train_encoded = encode_pairs(train_pairs, tokenizer, max_src_len, max_tgt_len)
    val_encoded = encode_pairs(val_pairs, tokenizer, max_src_len, max_tgt_len) if val_pairs else []

    # Create model
    transformer = Transformer(
        vocab_size=len(tokenizer.token_to_id),
        d_model=cfg.get("d_model", 256),
        num_heads=cfg.get("num_heads", 4),
        num_layers=cfg.get("num_layers", 3),
        d_ff=cfg.get("d_ff", 1024),
        max_seq_length=cfg.get("max_seq_length", 128),
    )

    # Trainer
    trainer = TransformerTrainer(
        transformer=transformer,
        learning_rate=cfg.get("learning_rate", 7e-4),
        optimizer="adam",
        clip_norm=1.0,
    )

    # Train
    history = trainer.train(
        train_data=train_encoded,
        val_data=val_encoded if val_encoded else None,
        epochs=cfg.get("epochs", 5),
        batch_size=cfg.get("batch_size", 32),
        save_attention=False,
    )

    # Save a final checkpoint (numpy-based weights)
    np.save(os.path.join(args.output_dir, "embedding.npy"), transformer.embedding)
    np.save(os.path.join(args.output_dir, "output_projection.npy"), transformer.output_projection)

    # Save config copy
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("✅ Training finished. Artifacts saved to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()


