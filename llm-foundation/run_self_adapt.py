#!/usr/bin/env python3
"""
Run SEAL-inspired self-adaptation on the NumPy Transformer with minimal compute.

This script:
- Adds `src` to PYTHONPATH
- Instantiates `CompleteTransformer`
- Creates a tiny dummy dataset (replace with your real data)
- Runs the outer adaptation loop and logs to adaptation_logs.json
"""

import os
import sys
import numpy as np


def _add_src_to_path() -> None:
    here = os.path.dirname(__file__)
    sys.path.append(os.path.join(here, "src"))


def make_dummy_pairs(vocab_size: int = 120, seq_len: int = 10, num_samples: int = 64):
    data = []
    for _ in range(num_samples):
        src = np.random.randint(1, vocab_size, size=(seq_len,))
        tgt = np.random.randint(1, vocab_size, size=(seq_len,))
        data.append((src, tgt))
    return data


def main() -> None:
    _add_src_to_path()

    # Lazy imports after path fix
    from transformer.complete.transformer import CompleteTransformer
    from transformer.training.adaptive_trainer import (
        AdaptiveTrainer,
        AdapterConfig,
        OuterLoopConfig,
    )

    # Repro seeds for dataset too
    np.random.seed(1234)

    # Instantiate a small transformer to keep compute light
    model = CompleteTransformer(
        vocab_size=120,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        max_seq_length=64,
        dropout=0.0,
    )

    # Few-shot dummy data; replace with your (src, tgt) pairs
    train_data = make_dummy_pairs(num_samples=64)
    val_data = make_dummy_pairs(num_samples=32)

    # Configure adapters and loop
    adapter_cfg = AdapterConfig(
        rank=4,
        scale=0.05,
        # Optionally narrow targets further to reduce compute
        # target_param_substrings=("attention_W_o", "output_projection"),
    )
    loop_cfg = OuterLoopConfig(
        num_iterations=5,
        candidates_per_iter=4,
        min_improvement=1e-4,
        seed=42,
        log_path="adaptation_logs.json",
        max_val_batch=4,
    )

    print("🚀 Running self-adaptation (SEAL-inspired)")
    print("=" * 60)

    adapter = AdaptiveTrainer(model, adapter_cfg, loop_cfg)
    adapter.run_outer_loop(train_data, val_data)

    print("\n✅ Completed. See adaptation_logs.json for decisions and metrics.")


if __name__ == "__main__":
    main()


