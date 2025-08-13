"""
Simple dataloader utilities for batching, padding, and shuffling.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple, Iterable

import numpy as np


class BatchIterator:
    """
    Minimal batch iterator over (src_ids, tgt_ids) pairs.
    Assumes inputs are already padded to fixed shapes.
    """

    def __init__(self, data: List[Tuple[np.ndarray, np.ndarray]], batch_size: int, shuffle: bool = True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start:start + self.batch_size]
            batch = [self.data[i] for i in batch_idx]
            src_batch = np.stack([b[0] for b in batch], axis=0)
            tgt_batch = np.stack([b[1] for b in batch], axis=0)
            yield src_batch, tgt_batch

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)


