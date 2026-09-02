import torch
import pytest

from lerobot.policies.smolvla.gradient_diagnostic import (
    GradientMetricAccumulator,
    compute_gradient_metrics,
    flatten_grads,
)


def test_flatten_grads_preserves_parameter_alignment_for_none():
    params = [torch.zeros(2), torch.zeros(3)]
    flat = flatten_grads([torch.tensor([1.0, 2.0]), None], params)
    assert torch.equal(flat, torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0]))


def test_metrics_use_raw_cosine_and_separate_lambda_ratios():
    params = [torch.zeros(2)]
    metrics = compute_gradient_metrics(
        [torch.tensor([1.0, 0.0])], [torch.tensor([2.0, 0.0])], params, 0.01, 0.05
    )
    assert metrics["grad_cosine"] == pytest.approx(1.0)
    assert metrics["raw_grad_ratio"] == pytest.approx(2.0)
    assert metrics["weighted_grad_ratio"] == pytest.approx(0.02)
    assert metrics["lambda_max_grad_ratio"] == pytest.approx(0.1)


def test_zero_norm_metrics_are_null():
    params = [torch.zeros(2)]
    metrics = compute_gradient_metrics([None], [None], params, 0.01, 0.05)
    assert metrics["grad_cosine"] is None
    assert metrics["conflict"] is None


def test_accumulator_excludes_invalid_cosines():
    acc = GradientMetricAccumulator()
    acc.add({"grad_cosine": -0.5, "conflict": True, "raw_grad_ratio": 2.0, "weighted_grad_ratio": 0.1, "lambda_max_grad_ratio": 0.2, "loss_depth": 0.3})
    acc.add({"grad_cosine": None, "conflict": None, "raw_grad_ratio": 0.0, "weighted_grad_ratio": 0.0, "lambda_max_grad_ratio": 0.0, "loss_depth": 0.1})
    summary = acc.summary()
    assert summary["n_measurements"] == 2
    assert summary["n_valid_gradient_measurements"] == 1
    assert summary["mean_cosine"] == -0.5
    assert summary["conflict_rate"] == 1.0
