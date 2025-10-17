#!/usr/bin/env python3
"""
Run SEAL-inspired self-adaptation on Amharic→Oromiffa translation pairs.

This hooks AdaptiveTrainer to the example vocabulary and pairs in
`src/transformer/examples/amharic_oromiffa_example.py` for a realistic
small-text scenario without heavy compute.
"""

import os
import sys
import numpy as np
from typing import List, Tuple


def _add_src_to_path() -> None:
    here = os.path.dirname(__file__)
    sys.path.append(os.path.join(here, "src"))


def build_dataset_from_examples(max_src_len: int = 12, max_tgt_len: int = 12) -> Tuple[dict, List[Tuple[np.ndarray, np.ndarray]]]:
    """Create (src, tgt) token pairs from the Amharic–Oromiffa example module.

    Returns:
        vocab: token->id dictionary
        data: list of (src_ids, tgt_ids) arrays padded to fixed lengths
    """
    from transformer.examples.amharic_oromiffa_example import (
        create_ethiopian_vocab,
        create_translation_pairs,
        text_to_sequence,
    )

    vocab = create_ethiopian_vocab()
    pairs = create_translation_pairs()

    data: List[Tuple[np.ndarray, np.ndarray]] = []
    for src_text, tgt_text in pairs:
        src_ids = np.array(text_to_sequence(src_text, vocab, max_length=max_src_len), dtype=int)
        tgt_ids = np.array(text_to_sequence(tgt_text, vocab, max_length=max_tgt_len), dtype=int)
        data.append((src_ids, tgt_ids))

    return vocab, data


def main() -> None:
    _add_src_to_path()

    # Lazy imports after path fix
    from transformer.complete.transformer import CompleteTransformer
    from transformer.training.adaptive_trainer import (
        AdaptiveTrainer,
        AdapterConfig,
        OuterLoopConfig,
    )

    # Repro seed
    np.random.seed(2025)

    # Build real-ish dataset from example pairs
    vocab, full_data = build_dataset_from_examples(max_src_len=12, max_tgt_len=12)

    # Split small train/val (e.g., 80/20)
    n = len(full_data)
    split = max(1, int(0.8 * n))
    train_data = full_data[:split]
    val_data = full_data[split:]
    if not val_data:
        val_data = full_data[-2:]

    # Instantiate compact transformer to keep compute light
    model = CompleteTransformer(
        vocab_size=len(vocab),
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        max_seq_length=64,
        dropout=0.0,
    )

    # Configure adapters and loop
    adapter_cfg = AdapterConfig(
        rank=4,
        scale=0.05,
        # Consider focusing on output head and O-proj for extra efficiency:
        # target_param_substrings=("attention_W_o", "output_projection"),
    )
    loop_cfg = OuterLoopConfig(
        num_iterations=8,
        candidates_per_iter=5,
        min_improvement=1e-4,
        seed=123,
        log_path="adaptation_logs.json",
        max_val_batch=4,
    )

    print("🇪🇹 Running self-adaptation on Amharic→Oromiffa pairs")
    print("=" * 60)
    print(f"Samples: train={len(train_data)}, val={len(val_data)}, vocab={len(vocab)}")

    adapter = AdaptiveTrainer(model, adapter_cfg, loop_cfg)
    adapter.run_outer_loop(train_data, val_data)

    print("\n✅ Completed. See adaptation_logs.json for decisions and metrics.")


if __name__ == "__main__":
    main()


