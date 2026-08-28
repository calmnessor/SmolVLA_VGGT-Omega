import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SpatialRegisterDepthProbe(nn.Module):
    def __init__(self, image_dim: int, register_dim: int, query_dim: int = 128):
        super().__init__()
        self.q_proj = nn.Linear(image_dim, query_dim)
        self.k_proj = nn.Linear(register_dim, query_dim)
        self.depth_head = nn.Sequential(nn.Linear(register_dim, query_dim), nn.GELU(), nn.Linear(query_dim, 1))

    def forward(self, image_tokens: Tensor, projected_registers: Tensor, patch_hw: tuple[int, int]) -> Tensor:
        if image_tokens.ndim != 3 or projected_registers.ndim != 3:
            raise ValueError("image_tokens and projected_registers must be [B, N, D]")
        if image_tokens.shape[1] != patch_hw[0] * patch_hw[1]:
            raise ValueError("patch_hw does not match image token count")
        query = self.q_proj(image_tokens.detach())
        key = self.k_proj(projected_registers)
        weights = torch.softmax(query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1]), dim=-1)
        attended = weights @ projected_registers
        return (F.softplus(self.depth_head(attended).squeeze(-1)) + 1e-6).reshape(image_tokens.shape[0], *patch_hw)

    def forward_views(self, image_tokens: Tensor, projected_registers: Tensor, patch_hw: tuple[int, int]) -> Tensor:
        if image_tokens.ndim != 4 or projected_registers.ndim != 4 or image_tokens.shape[:2] != projected_registers.shape[:2]:
            raise ValueError("view tensors must have matching [B, V, ...] shape")
        return torch.stack([self(image_tokens[:, v], projected_registers[:, v], patch_hw) for v in range(image_tokens.shape[1])], dim=1)
