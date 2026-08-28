import math

import torch

from lerobot.policies.smolvla.depth_distillation import (
    confidence_masked_log_l1,
    depth_lambda,
    pool_teacher_to_patch_grid,
)
from lerobot.policies.smolvla.depth_probe import SpatialRegisterDepthProbe


def test_depth_lambda_uses_linear_warmup():
    assert depth_lambda(0, 0.05, 1000) == 0.0
    assert depth_lambda(500, 0.05, 1000) == 0.025
    assert depth_lambda(1000, 0.05, 1000) == 0.05
    assert depth_lambda(2000, 0.05, 1000) == 0.05


def test_pool_teacher_filters_invalid_pixels_and_low_confidence():
    depth = torch.tensor(
        [[[1.0, 2.0, 10.0, 10.0], [3.0, 4.0, 10.0, 10.0],
          [20.0, 20.0, float("nan"), -1.0], [20.0, 20.0, 8.0, 9.0]]]
    )
    confidence = torch.tensor(
        [[[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],
          [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, float("inf"), 1.0]]]
    )
    image_valid = torch.ones_like(depth, dtype=torch.bool)
    image_valid[:, 2:, :2] = False

    pooled, mask = pool_teacher_to_patch_grid(
        depth,
        confidence,
        image_valid,
        out_hw=(2, 2),
        confidence_quantile=0.2,
        min_valid_ratio=0.25,
    )

    assert pooled.shape == (1, 2, 2)
    assert mask.tolist() == [[[True, True], [False, True]]]
    assert torch.isclose(pooled[0, 0, 0], torch.tensor(3.0))
    assert torch.isclose(pooled[0, 0, 1], torch.tensor(10.0))
    assert torch.isclose(pooled[0, 1, 1], torch.tensor(9.0))


def test_pool_teacher_rejects_patch_below_valid_ratio():
    depth = torch.ones(1, 4, 4)
    confidence = torch.ones_like(depth)
    image_valid = torch.zeros_like(depth, dtype=torch.bool)
    image_valid[0, 0, 0] = True

    _, mask = pool_teacher_to_patch_grid(
        depth, confidence, image_valid, out_hw=(1, 1), min_valid_ratio=0.26
    )

    assert not mask.item()


def test_log_l1_normalizes_each_camera_before_averaging():
    student = [torch.tensor([[[1.0, math.e], [1.0, 1.0]]]), torch.ones(1, 1, 1)]
    teacher = [torch.ones(1, 2, 2), torch.full((1, 1, 1), math.e**2)]
    masks = [torch.tensor([[[True, True], [False, False]]]), torch.ones(1, 1, 1, dtype=torch.bool)]

    loss = confidence_masked_log_l1(student, teacher, masks)

    assert torch.isclose(loss, torch.tensor(1.25), atol=1e-6)


def test_log_l1_returns_finite_zero_when_no_patch_is_valid():
    student = torch.ones(1, 2, 2, requires_grad=True)
    loss = confidence_masked_log_l1(student, torch.ones_like(student), torch.zeros_like(student, dtype=torch.bool))

    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_depth_probe_is_positive_dropout_free_and_detaches_image_queries():
    probe = SpatialRegisterDepthProbe(image_dim=6, register_dim=8, query_dim=4)
    image_tokens = torch.randn(2, 4, 6, requires_grad=True)
    registers = torch.randn(2, 3, 8, requires_grad=True)

    depth = probe(image_tokens, registers, patch_hw=(2, 2))
    depth.sum().backward()

    assert depth.shape == (2, 2, 2)
    assert torch.all(depth > 0)
    assert image_tokens.grad is None
    assert registers.grad is not None
    assert not any(isinstance(module, torch.nn.Dropout) for module in probe.modules())


def test_depth_probe_keeps_camera_registers_separate():
    torch.manual_seed(0)
    probe = SpatialRegisterDepthProbe(image_dim=4, register_dim=4, query_dim=4)
    images = torch.randn(1, 2, 4, 4)
    registers = torch.randn(1, 2, 3, 4)

    both = probe.forward_views(images, registers, patch_hw=(2, 2))
    changed = registers.clone()
    changed[:, 1] += 100
    changed_both = probe.forward_views(images, changed, patch_hw=(2, 2))

    assert torch.equal(both[:, 0], changed_both[:, 0])
    assert not torch.equal(both[:, 1], changed_both[:, 1])
