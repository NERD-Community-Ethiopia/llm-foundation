#!/usr/bin/env python3
"""
Self-adapt the NumPy Transformer from a TSV parallel corpus (real-world flow).

Input format (tab-separated):
    source_text<TAB>target_text

Example usage (from repo root):
    python run_self_adapt_from_tsv.py --tsv data/domain_pairs.tsv \
        --max_src_len 64 --max_tgt_len 64 --limit 200 \
        --iterations 10 --candidates 6
"""

import os
import sys
import csv
import argparse
import numpy as np
from typing import List, Tuple, Dict


def _add_src_to_path() -> None:
    here = os.path.dirname(__file__)
    sys.path.append(os.path.join(here, "src"))


def load_pairs_tsv(path: str, limit: int = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row or len(row) < 2:
                continue
            s = row[0].strip()
            t = row[1].strip()
            if s and t:
                pairs.append((s, t))
            if limit and len(pairs) >= limit:
                break
    return pairs


def build_vocab_from_corpus(pairs: List[Tuple[str, str]], min_freq: int = 1) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for s, t in pairs:
        for tok in s.split():
            freq[tok] = freq.get(tok, 0) + 1
        for tok in t.split():
            freq[tok] = freq.get(tok, 0) + 1
    vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
    for w, c in freq.items():
        if c >= min_freq and w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def text_to_ids(text: str, vocab: Dict[str, int], max_len: int) -> np.ndarray:
    ids = [vocab.get(tok, vocab['<UNK>']) for tok in text.split()]
    ids = ids[:max_len] + [vocab['<PAD>']] * max(0, max_len - len(ids))
    return np.array(ids, dtype=int)


def build_dataset(
    tsv_path: str,
    max_src_len: int,
    max_tgt_len: int,
    limit: int = None,
    min_freq: int = 1,
) -> Tuple[Dict[str, int], List[Tuple[np.ndarray, np.ndarray]]]:
    raw_pairs = load_pairs_tsv(tsv_path, limit=limit)
    vocab = build_vocab_from_corpus(raw_pairs, min_freq=min_freq)
    data: List[Tuple[np.ndarray, np.ndarray]] = []
    for s, t in raw_pairs:
        src_ids = text_to_ids(s, vocab, max_src_len)
        tgt_ids = text_to_ids(t, vocab, max_tgt_len)
        data.append((src_ids, tgt_ids))
    return vocab, data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True, help="Path to TSV with source<TAB>target")
    p.add_argument("--max_src_len", type=int, default=64)
    p.add_argument("--max_tgt_len", type=int, default=64)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--candidates", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log", default="adaptation_logs.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _add_src_to_path()

    # Deferred imports after path setup
    from transformer.complete.transformer import CompleteTransformer
    from transformer.training.adaptive_trainer import (
        AdaptiveTrainer,
        AdapterConfig,
        OuterLoopConfig,
    )

    # Reproducibility
    np.random.seed(args.seed)

    # Build dataset from TSV
    vocab, data = build_dataset(
        tsv_path=args.tsv,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        limit=args.limit,
        min_freq=1,
    )

    if not data:
        raise SystemExit("No valid pairs loaded from TSV.")

    # Train/val split (80/20)
    n = len(data)
    split = max(1, int(0.8 * n))
    train_data = data[:split]
    val_data = data[split:] or data[-2:]

    # Compact transformer for minimal compute
    model = CompleteTransformer(
        vocab_size=len(vocab),
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        max_seq_length=max(args.max_src_len, args.max_tgt_len),
        dropout=0.0,
    )

    adapter_cfg = AdapterConfig(
        rank=4,
        scale=0.05,
        # Optionally narrow target params to reduce compute further:
        # target_param_substrings=("attention_W_o", "output_projection"),
    )
    loop_cfg = OuterLoopConfig(
        num_iterations=args.iterations,
        candidates_per_iter=args.candidates,
        min_improvement=1e-4,
        seed=args.seed,
        log_path=args.log,
        max_val_batch=6,
    )

    print("📁 TSV:", args.tsv)
    print("🔤 Vocab size:", len(vocab))
    print("📊 Samples: train=", len(train_data), "val=", len(val_data))
    print("🚀 Running self-adaptation...")
    print("=" * 60)

    adapter = AdaptiveTrainer(model, adapter_cfg, loop_cfg)
    adapter.run_outer_loop(train_data, val_data)

    print("\n✅ Completed. See", args.log, "for decisions and metrics.")


if __name__ == "__main__":
    main()


