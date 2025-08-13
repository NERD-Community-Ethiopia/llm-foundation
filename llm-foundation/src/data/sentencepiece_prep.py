"""
SentencePiece training and application helpers.

Requires: pip install sentencepiece
"""

from __future__ import annotations

from typing import Iterable, List
import os
import sentencepiece as spm


def write_corpus_for_spm(lines: Iterable[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")


def train_sentencepiece(
    input_path: str,
    model_prefix: str,
    vocab_size: int = 32000,
    model_type: str = "unigram",
    character_coverage: float = 1.0,
    user_defined_symbols: List[str] | None = None,
) -> None:
    user_symbols = user_defined_symbols or []
    spm.SentencePieceTrainer.Train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        user_defined_symbols=",".join(user_symbols) if user_symbols else "",
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_id=0,
    )


class SentencePieceTokenizer:
    def __init__(self, model_file: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            # Rely on BOS/EOS ids configured during training
            ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return self.sp.decode(list(ids))


