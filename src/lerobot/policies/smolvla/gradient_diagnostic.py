"""Utilities for measuring action/depth gradient interaction."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import torch


def flatten_grads(grads: Iterable[torch.Tensor | None], params: Iterable[torch.Tensor]) -> torch.Tensor:
    """Flatten gradients while preserving parameter positions, filling missing grads with zeros."""
    flat = []
    for grad, param in zip(grads, params, strict=True):
        if grad is None:
            flat.append(torch.zeros_like(param, dtype=torch.float32).reshape(-1))
        else:
            flat.append(grad.detach().float().reshape(-1))
    if not flat:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(flat)


def compute_gradient_metrics(
    action_grads: Iterable[torch.Tensor | None],
    depth_grads: Iterable[torch.Tensor | None],
    params: Iterable[torch.Tensor],
    current_lambda: float,
    lambda_max: float,
    eps: float = 1e-8,
) -> dict:
    """Compute raw alignment and lambda-scaled depth influence metrics."""
    params = list(params)
    action = flatten_grads(action_grads, params)
    depth = flatten_grads(depth_grads, params)
    action_norm = float(action.norm().item())
    depth_norm = float(depth.norm().item())
    raw_ratio = depth_norm / (action_norm + eps)
    result = {
        "action_grad_norm": action_norm,
        "depth_grad_norm_raw": depth_norm,
        "raw_grad_ratio": raw_ratio,
        "weighted_grad_ratio": float(current_lambda) * raw_ratio,
        "lambda_max_grad_ratio": float(lambda_max) * raw_ratio,
    }
    if action_norm < eps or depth_norm < eps:
        result["grad_cosine"] = None
        result["conflict"] = None
    else:
        cosine = float(torch.dot(action, depth).item() / (action_norm * depth_norm + eps))
        result["grad_cosine"] = max(-1.0, min(1.0, cosine))
        result["conflict"] = cosine < 0.0
    return result


class GradientMetricAccumulator:
    """Accumulate JSON-serializable diagnostic records and summarize them."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def add(self, record: dict) -> None:
        self.records.append(dict(record))

    @staticmethod
    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "median": None, "p10": None, "p95": None}
        values = sorted(values)

        def percentile(q: float) -> float:
            index = (len(values) - 1) * q
            lower, upper = math.floor(index), math.ceil(index)
            if lower == upper:
                return values[lower]
            return values[lower] + (values[upper] - values[lower]) * (index - lower)

        return {
            "mean": sum(values) / len(values),
            "median": percentile(0.5),
            "p10": percentile(0.1),
            "p95": percentile(0.95),
        }

    def summary(self) -> dict:
        valid = [r for r in self.records if r.get("grad_cosine") is not None]
        cosine_stats = self._stats([float(r["grad_cosine"]) for r in valid])
        raw_stats = self._stats([float(r.get("raw_grad_ratio", 0.0)) for r in self.records])
        weighted_stats = self._stats([float(r.get("weighted_grad_ratio", 0.0)) for r in self.records])
        max_stats = self._stats([float(r.get("lambda_max_grad_ratio", 0.0)) for r in self.records])
        depth_losses = [float(r["loss_depth"]) for r in self.records if r.get("loss_depth") is not None]
        valid_ratios = [float(r["valid_depth_patch_ratio"]) for r in self.records if r.get("valid_depth_patch_ratio") is not None]
        return {
            "n_measurements": len(self.records),
            "n_valid_gradient_measurements": len(valid),
            "mean_cosine": cosine_stats["mean"],
            "median_cosine": cosine_stats["median"],
            "p10_cosine": cosine_stats["p10"],
            "conflict_rate": (sum(bool(r["conflict"]) for r in valid) / len(valid)) if valid else None,
            "mean_raw_ratio": raw_stats["mean"],
            "mean_weighted_ratio": weighted_stats["mean"],
            "p95_weighted_ratio": weighted_stats["p95"],
            "mean_lambda_max_ratio": max_stats["mean"],
            "p95_lambda_max_ratio": max_stats["p95"],
            "mean_depth_loss": (sum(depth_losses) / len(depth_losses)) if depth_losses else None,
            "mean_valid_depth_patch_ratio": (sum(valid_ratios) / len(valid_ratios)) if valid_ratios else None,
        }
