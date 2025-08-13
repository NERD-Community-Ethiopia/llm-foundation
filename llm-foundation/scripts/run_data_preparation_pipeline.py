#!/usr/bin/env python3
"""
Run the Amharic→Oromiffa data preparation pipeline end-to-end.

This is a thin orchestrator around scripts/prepare_am_om_data.py.

Examples:
  python scripts/run_data_preparation_pipeline.py \
    --inputs data/raw/*.tsv \
    --out-dir data/processed/am-om \
    --spm --vocab-size 32000
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="+",
        default=["data/raw/*.tsv"],
        help="One or more TSV globs with am\tom pairs",
    )
    p.add_argument(
        "--out-dir",
        default="data/processed/am-om",
        help="Output directory for cleaned splits and tokenizer",
    )
    p.add_argument("--spm", action="store_true", help="Train SentencePiece on train split")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--model-type", default="unigram", choices=["unigram", "bpe", "char", "word"])
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--length-ratio", type=float, default=3.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve input files from globs
    input_files: List[str] = []
    for pattern in args.inputs:
        input_files.extend(glob.glob(pattern))
    input_files = sorted(set(input_files))

    if not input_files:
        print("❌ No input TSV files found. Check --inputs globs.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # Build command to invoke the underlying prep script
    script_path = os.path.join(os.path.dirname(__file__), "prepare_am_om_data.py")
    cmd = [
        sys.executable,
        script_path,
        "--input",
        *input_files,
        "--out-dir",
        args.out_dir,
        "--val-ratio",
        str(args.val_ratio),
        "--test-ratio",
        str(args.test_ratio),
        "--max-len",
        str(args.max_len),
        "--length-ratio",
        str(args.length_ratio),
    ]
    if args.spm:
        cmd.extend(["--spm", "--vocab-size", str(args.vocab_size), "--model-type", args.model_type])

    print("🚀 Running data preparation...")
    print(" ", " ".join(cmd))
    subprocess.check_call(cmd)

    # Summarize outputs
    print("\n📦 Artifacts in:", os.path.abspath(args.out_dir))
    for name in ["train.tsv", "val.tsv", "test.tsv", "train.txt", "am_om_sp.model", "am_om_sp.vocab"]:
        path = os.path.join(args.out_dir, name)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024.0
            print(f"  - {name} ({size_kb:.1f} KB)")

    print("\n✅ Data preparation pipeline completed.")


if __name__ == "__main__":
    main()


