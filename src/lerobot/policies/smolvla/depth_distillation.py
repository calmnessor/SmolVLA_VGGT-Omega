from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def depth_lambda(step: int, lambda_max: float, warmup_steps: int = 1000) -> float:
    if lambda_max < 0:
        raise ValueError("lambda_max must be non-negative")
    if warmup_steps <= 0:
        return float(lambda_max)
    return float(lambda_max) * min(max(step, 0) / warmup_steps, 1.0)


def pool_teacher_to_patch_grid(depth: Tensor, confidence: Tensor, image_valid_mask: Tensor, out_hw: tuple[int, int], confidence_quantile: float = 0.2, min_valid_ratio: float = 0.25) -> tuple[Tensor, Tensor]:
    if depth.ndim != 3 or confidence.shape != depth.shape or image_valid_mask.shape != depth.shape:
        raise ValueError("depth, confidence, and image_valid_mask must have matching [B, H, W] shapes")
    finite_conf = torch.isfinite(confidence) & image_valid_mask.bool()
    thresholds = []
    for i in range(depth.shape[0]):
        values = confidence[i][finite_conf[i]]
        thresholds.append(values.quantile(confidence_quantile) if values.numel() else confidence.new_tensor(float("inf")))
    threshold = torch.stack(thresholds)[:, None, None]
    valid = image_valid_mask.bool() & torch.isfinite(depth) & (depth > 0) & torch.isfinite(confidence) & (confidence >= threshold)
    valid_fraction = F.adaptive_avg_pool2d(valid.to(depth.dtype)[:, None], out_hw)[:, 0]
    depth_sum = F.adaptive_avg_pool2d(depth.masked_fill(~valid, 0)[:, None], out_hw)[:, 0]
    pooled = depth_sum / valid_fraction.clamp_min(torch.finfo(depth.dtype).eps)
    patch_valid = valid_fraction >= min_valid_ratio
    return pooled.masked_fill(~patch_valid, 0), patch_valid


def confidence_masked_log_l1(student_depth: Tensor | Sequence[Tensor], teacher_depth: Tensor | Sequence[Tensor], valid_mask: Tensor | Sequence[Tensor], eps: float = 1e-6) -> Tensor:
    if isinstance(student_depth, Tensor):
        students, teachers, masks = [student_depth], [teacher_depth], [valid_mask]
    else:
        students, teachers, masks = list(student_depth), list(teacher_depth), list(valid_mask)
    if not (len(students) == len(teachers) == len(masks)) or not students:
        raise ValueError("camera lists must have the same non-zero length")
    losses = []
    for student, teacher, mask in zip(students, teachers, masks, strict=True):
        per_patch = (torch.log(student.clamp_min(eps)) - torch.log(teacher.clamp_min(eps))).abs()
        losses.append((per_patch * mask).sum() / mask.sum().clamp_min(1))
    return torch.stack(losses).mean()
