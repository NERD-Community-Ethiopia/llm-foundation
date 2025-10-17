"""
Self-Adaptive Training (SEAL-inspired) for the custom NumPy Transformer.

This module implements a lightweight inner/outer loop that proposes and evaluates
small, low-rank parameter edits (adapters) on top of existing Transformer weights.

Design goals:
- Keep GPU/compute minimal by only updating low-rank adapter factors for a subset
  of parameters (e.g., attention projections and output heads).
- Avoid full retraining; use few-shot batches and small gradient-free or
  gradient-lite edits.
- Maintain reproducibility by setting seeds and versioning accepted edits.
- Log every iteration and decision to a JSONL file: adaptation_logs.json

Usage pattern (high level):
    adapter = AdaptiveTrainer(transformer)
    adapter.run_outer_loop(train_data, val_data)

Notes:
- This integrates non-invasively by temporarily applying low-rank deltas directly
  to numpy arrays, then reverting, committing only when an edit is accepted.
- No changes are required in the existing forward pass implementations.
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer random seed to apply across numpy and Python.
    """
    np.random.seed(seed)


def _now_ts() -> float:
    return float(time.time())


def _hash_edit(payload: Dict[str, Any]) -> str:
    """Create a short stable identifier for an edit proposal or decision."""
    m = hashlib.sha256()
    m.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    return m.hexdigest()[:10]


@dataclass
class AdapterConfig:
    """Configuration for low-rank adapters.

    Attributes:
        rank: Low rank R for the factorization delta = alpha * A @ B.
        scale: Scaling factor alpha for how strongly edits affect the base weights.
        target_param_substrings: Only parameters whose names contain any of these
            substrings will be adapted.
    """

    rank: int = 4
    scale: float = 0.05
    target_param_substrings: Tuple[str, ...] = (
        "attention_W_q",
        "attention_W_k",
        "attention_W_v",
        "attention_W_o",
        "output_projection",
    )


@dataclass
class OuterLoopConfig:
    """Top-level control for adaptation cycles."""

    num_iterations: int = 10
    candidates_per_iter: int = 4
    min_improvement: float = 1e-4
    seed: int = 42
    log_path: str = "adaptation_logs.json"
    max_val_batch: int = 8  # evaluate on a few small batches for speed


class LowRankAdapter:
    """Low-rank adapter applied to a specific weight matrix W.

    We represent an additive delta as: delta = alpha * A @ B, where
    A in R^{d_out x r}, B in R^{r x d_in}, r is small rank.
    """

    def __init__(self, param: np.ndarray, rank: int, alpha: float):
        self.param = param
        self.rank = rank
        self.alpha = alpha

        d_out, d_in = param.shape
        # Initialize small factors near zero for stability
        self.A = np.random.randn(d_out, rank) * 1e-3
        self.B = np.random.randn(rank, d_in) * 1e-3

        # Working buffers to avoid reallocations
        self._delta = np.zeros_like(param)

    def compute_delta(self) -> np.ndarray:
        np.dot(self.A, self.B, out=self._delta)
        return self.alpha * self._delta

    def apply(self) -> None:
        self.param += self.compute_delta()

    def revert(self) -> None:
        self.param -= self.compute_delta()

    def perturb(self, std: float) -> Tuple[np.ndarray, np.ndarray]:
        """Add a random Gaussian proposal to A and B and return the noise."""
        noise_A = np.random.randn(*self.A.shape) * std
        noise_B = np.random.randn(*self.B.shape) * std
        self.A += noise_A
        self.B += noise_B
        return noise_A, noise_B

    def unperturb(self, noise_A: np.ndarray, noise_B: np.ndarray) -> None:
        self.A -= noise_A
        self.B -= noise_B


class AdaptiveTrainer:
    """SEAL-inspired self-adaptation for the NumPy Transformer.

    This class wraps a provided transformer instance that follows the internal
    parameter conventions used in `training_pipeline.py` and updates only
    low-rank adapters on a subset of parameters.
    """

    def __init__(
        self,
        transformer,
        adapter_cfg: Optional[AdapterConfig] = None,
        loop_cfg: Optional[OuterLoopConfig] = None,
    ) -> None:
        self.transformer = transformer
        self.adapter_cfg = adapter_cfg or AdapterConfig()
        self.loop_cfg = loop_cfg or OuterLoopConfig()

        set_global_seed(self.loop_cfg.seed)

        # Map of parameter-name -> adapter
        self.adapters: Dict[str, LowRankAdapter] = {}
        self._index_parameters()

        # Logging
        self.log_path = self.loop_cfg.log_path
        self._ensure_log_file()

    # ---------- Public API ----------
    def run_outer_loop(
        self,
        train_data: List[Tuple[np.ndarray, np.ndarray]],
        val_data: List[Tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Run the propose-evaluate-accept loop for several iterations.

        - Propose several candidate edit bundles per iteration (inner loop)
        - Evaluate quickly on a small slice of validation data
        - Accept if best candidate improves the metric beyond a threshold
        - Log decisions and maintain an edit history
        """

        baseline = self._evaluate_quick(val_data)
        self._log_event(
            event_type="baseline_eval",
            payload={"baseline_loss": baseline},
        )

        for iteration in range(self.loop_cfg.num_iterations):
            candidates = []
            # Generate candidate edits (self-edits and possible data augmentation notes)
            for _ in range(self.loop_cfg.candidates_per_iter):
                cand = self._propose_candidate_edit(train_data)
                candidates.append(cand)

            # Evaluate candidates
            scores = []
            for cand in candidates:
                self._apply_candidate(cand)
                loss = self._evaluate_quick(val_data)
                self._revert_candidate(cand)
                scores.append(loss)

            best_idx = int(np.argmin(np.array(scores)))
            best_loss = scores[best_idx]

            accepted = (baseline - best_loss) >= self.loop_cfg.min_improvement
            decision = {
                "iteration": iteration,
                "baseline_loss": baseline,
                "best_loss": best_loss,
                "improvement": float(baseline - best_loss),
                "accepted": bool(accepted),
                "num_candidates": len(candidates),
            }

            if accepted:
                # Commit the best candidate
                self._apply_candidate(candidates[best_idx])
                baseline = best_loss
                decision["committed_candidate_id"] = candidates[best_idx]["id"]
            else:
                decision["committed_candidate_id"] = None

            self._log_event(event_type="outer_iter_decision", payload=decision)

    # ---------- Internal: parameter indexing ----------
    def _index_parameters(self) -> None:
        """Identify and wrap target parameters with adapters.

        We mirror the naming style used by `TransformerTrainer._get_parameters`.
        """
        param_list = self._get_parameters_like_trainer()
        for name, param in param_list:
            if not isinstance(param, np.ndarray):
                continue
            if param.ndim != 2:
                continue
            if not any(s in name for s in self.adapter_cfg.target_param_substrings):
                continue

            self.adapters[name] = LowRankAdapter(
                param=param, rank=self.adapter_cfg.rank, alpha=self.adapter_cfg.scale
            )

    def _get_parameters_like_trainer(self) -> List[Tuple[str, np.ndarray]]:
        """Traverse the provided transformer to extract named weight arrays.

        This follows the same structure expected by `training_pipeline.TransformerTrainer`.
        """
        params: List[Tuple[str, np.ndarray]] = []

        # Embedding and output projection (if present)
        if hasattr(self.transformer, "embedding"):
            params.append(("embedding", self.transformer.embedding))
        if hasattr(self.transformer, "output_projection"):
            params.append(("output_projection", self.transformer.output_projection))

        # Encoder layers
        if hasattr(self.transformer, "encoder_layers"):
            for i, layer in enumerate(self.transformer.encoder_layers):
                if hasattr(layer, "self_attention"):
                    attn = layer.self_attention
                    if hasattr(attn, "W_q"):
                        params.append((f"encoder_{i}_attention_W_q", attn.W_q))
                    if hasattr(attn, "W_k"):
                        params.append((f"encoder_{i}_attention_W_k", attn.W_k))
                    if hasattr(attn, "W_v"):
                        params.append((f"encoder_{i}_attention_W_v", attn.W_v))
                    if hasattr(attn, "W_o"):
                        params.append((f"encoder_{i}_attention_W_o", attn.W_o))
                if hasattr(layer, "ff_network"):
                    ff = layer.ff_network
                    if hasattr(ff, "W1"):
                        params.append((f"encoder_{i}_ff_W1", ff.W1))
                    if hasattr(ff, "W2"):
                        params.append((f"encoder_{i}_ff_W2", ff.W2))
                    if hasattr(ff, "b1"):
                        params.append((f"encoder_{i}_ff_b1", ff.b1))
                    if hasattr(ff, "b2"):
                        params.append((f"encoder_{i}_ff_b2", ff.b2))

        # Decoder layers
        if hasattr(self.transformer, "decoder_layers"):
            for i, layer in enumerate(self.transformer.decoder_layers):
                if hasattr(layer, "self_attention"):
                    attn = layer.self_attention
                    if hasattr(attn, "W_q"):
                        params.append((f"decoder_{i}_self_attention_W_q", attn.W_q))
                    if hasattr(attn, "W_k"):
                        params.append((f"decoder_{i}_self_attention_W_k", attn.W_k))
                    if hasattr(attn, "W_v"):
                        params.append((f"decoder_{i}_self_attention_W_v", attn.W_v))
                    if hasattr(attn, "W_o"):
                        params.append((f"decoder_{i}_self_attention_W_o", attn.W_o))
                if hasattr(layer, "cross_attention"):
                    attn = layer.cross_attention
                    if hasattr(attn, "W_q"):
                        params.append((f"decoder_{i}_cross_attention_W_q", attn.W_q))
                    if hasattr(attn, "W_k"):
                        params.append((f"decoder_{i}_cross_attention_W_k", attn.W_k))
                    if hasattr(attn, "W_v"):
                        params.append((f"decoder_{i}_cross_attention_W_v", attn.W_v))
                    if hasattr(attn, "W_o"):
                        params.append((f"decoder_{i}_cross_attention_W_o", attn.W_o))
                if hasattr(layer, "ff_network"):
                    ff = layer.ff_network
                    if hasattr(ff, "W1"):
                        params.append((f"decoder_{i}_ff_W1", ff.W1))
                    if hasattr(ff, "W2"):
                        params.append((f"decoder_{i}_ff_W2", ff.W2))
                    if hasattr(ff, "b1"):
                        params.append((f"decoder_{i}_ff_b1", ff.b1))
                    if hasattr(ff, "b2"):
                        params.append((f"decoder_{i}_ff_b2", ff.b2))

        return params

    # ---------- Internal: edit proposals ----------
    def _propose_candidate_edit(self, train_data) -> Dict[str, Any]:
        """Create a bundle of adapter perturbations and optional augmentation note.

        This uses a gradient-free Gaussian perturbation to propose changes.
        """
        # Select a random subset of adapters to modify
        all_names = list(self.adapters.keys())
        if not all_names:
            raise RuntimeError("No adapters indexed; cannot propose edits.")

        subset_size = max(1, int(0.25 * len(all_names)))
        chosen = list(np.random.choice(all_names, size=subset_size, replace=False))

        # Propose small noises
        std = 1e-3
        noises: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name in chosen:
            noises[name] = self.adapters[name].perturb(std)

        # Optionally suggest simple text/data augmentation idea (metadata only)
        aug_note = self._suggest_augmentation_note(train_data)

        cand = {
            "id": _hash_edit({"chosen": chosen, "time": _now_ts()}),
            "chosen_params": chosen,
            "noises": {k: (v[0].shape, v[1].shape) for k, v in noises.items()},
            "_noises_raw": noises,  # kept in-memory only for apply/revert
            "augmentation_note": aug_note,
        }

        # Revert immediately; application is explicit in evaluation/commit
        for name in chosen:
            noise_A, noise_B = noises[name]
            self.adapters[name].unperturb(noise_A, noise_B)

        self._log_event(event_type="candidate_proposed", payload={
            "candidate_id": cand["id"],
            "chosen_params": chosen,
            "augmentation_note": aug_note,
        })
        return cand

    def _apply_candidate(self, cand: Dict[str, Any]) -> None:
        # Re-apply the stored noise and then apply deltas to params
        for name in cand["chosen_params"]:
            noise_A, noise_B = cand["_noises_raw"][name]
            self.adapters[name].A += noise_A
            self.adapters[name].B += noise_B
        for name in cand["chosen_params"]:
            self.adapters[name].apply()

    def _revert_candidate(self, cand: Dict[str, Any]) -> None:
        # Revert deltas and remove noise from factors
        for name in cand["chosen_params"]:
            self.adapters[name].revert()
        for name in cand["chosen_params"]:
            noise_A, noise_B = cand["_noises_raw"][name]
            self.adapters[name].A -= noise_A
            self.adapters[name].B -= noise_B

    def _suggest_augmentation_note(self, train_data) -> Optional[str]:
        """Heuristic suggestions for self-edit data augmentation.

        This returns metadata notes only; implementers can optionally act on it
        elsewhere without changing core training loops here.
        """
        suggestions = [
            "augment:random_token_dropout(p=0.05)",
            "augment:swap_adjacent_tokens(p=0.02)",
            "augment:mask_rare_tokens(p=0.03)",
            None,
        ]
        return np.random.choice(suggestions)

    # ---------- Internal: fast evaluation ----------
    def _evaluate_quick(self, val_data: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Compute a quick average loss on a few small batches for selection.

        We use the same cross-entropy as in the training pipeline, but only a
        handful of batches for speed and minimal compute.
        """
        if not val_data:
            return float("inf")

        # Sample a slice of validation data
        batch_size = 4
        max_batches = max(1, self.loop_cfg.max_val_batch)
        indices = np.random.permutation(len(val_data))[: max_batches * batch_size]
        total_loss = 0.0
        steps = 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch = [val_data[j] for j in batch_idx]
            src_batch = np.array([item[0] for item in batch])
            tgt_batch = np.array([item[1] for item in batch])

            logits = self.transformer.forward(src_batch, tgt_batch)
            loss = self._cross_entropy_loss_only(logits, tgt_batch)
            total_loss += loss
            steps += 1

        return float(total_loss / max(1, steps))

    @staticmethod
    def _cross_entropy_loss_only(logits: np.ndarray, targets: np.ndarray, ignore_index: int = 0) -> float:
        # Simplified CE loss mirroring training_pipeline logic
        batch_size, seq_length, vocab_size = logits.shape
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        loss = 0.0
        count = 0
        for b in range(batch_size):
            for t in range(seq_length):
                target_token = int(targets[b, t])
                if target_token == ignore_index:
                    continue
                loss -= np.log(probs[b, t, target_token] + 1e-8)
                count += 1
        if count == 0:
            return 0.0
        return loss / count

    # ---------- Internal: logging ----------
    def _ensure_log_file(self) -> None:
        # Use JSONL for append-only logging
        log_dir = os.path.dirname(self.log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        # Touch file
        with open(self.log_path, "a", encoding="utf-8") as _:
            pass

    def _log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        record = {
            "t": _now_ts(),
            "event": event_type,
            "payload": payload,
            "version": "v1",
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


