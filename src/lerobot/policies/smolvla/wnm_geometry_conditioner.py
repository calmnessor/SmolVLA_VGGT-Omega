from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def prepare_vggt_history(history: Tensor, resolution: int) -> Tensor:
    if history.ndim != 5 or history.shape[2] != 3:
        raise ValueError(f"Expected history [B,T,3,H,W], got {tuple(history.shape)}")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    x = history.float().clamp(0, 1)
    b, t, c, h, w = x.shape
    x = F.interpolate(x.reshape(b * t, c, h, w), size=(resolution, resolution), mode="bicubic", align_corners=False)
    return x.reshape(b, t, c, resolution, resolution)


class WNMGeometryConditioner(nn.Module):
    def __init__(self, config, output_dim: int, aggregator: nn.Module | None = None, adapter: nn.Module | None = None):
        super().__init__()
        self.config = config
        if aggregator is None:
            if not config.wnm_geometry_code_path:
                raise ValueError("wnm_geometry_code_path is required for production WNM geometry")
            root = str(Path(config.wnm_geometry_code_path).expanduser())
            if root not in sys.path:
                sys.path.insert(0, root)
            Aggregator = importlib.import_module("wnm_3d.modules.vggt_omega.models.aggregator").Aggregator
            aggregator = Aggregator(patch_size=config.wnm_geometry_patch_size)
            if config.wnm_geometry_checkpoint:
                state = torch.load(config.wnm_geometry_checkpoint, map_location="cpu")
                aggregator.load_state_dict(state.get("state_dict", state), strict=False)
        self.__dict__["_aggregator"] = aggregator
        self._freeze_aggregator()
        if adapter is None:
            Adapter = importlib.import_module("wnm_3d.modules.vggt_geometry_adapter").VGGTOmegaGeometryAdapter
            tt, hh, ww = config.wnm_geometry_target_grid
            adapter = Adapter(output_dim=output_dim, adapter_dim=config.wnm_geometry_adapter_dim,
                              num_heads=config.wnm_geometry_adapter_heads, num_blocks=config.wnm_geometry_adapter_blocks,
                              source_t=config.wnm_geometry_history, source_h=config.wnm_geometry_resolution // config.wnm_geometry_patch_size,
                              source_w=config.wnm_geometry_resolution // config.wnm_geometry_patch_size,
                              target_t=tt, target_h=hh, target_w=ww)
        self.adapter = adapter

    def _freeze_aggregator(self):
        self._aggregator.eval()
        for p in self._aggregator.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._aggregator.eval()
        return self

    def forward(self, history: Tensor) -> Tensor:
        if history.ndim != 5 or history.shape[1] != self.config.wnm_geometry_history:
            raise ValueError(f"Expected {self.config.wnm_geometry_history} history frames, got {tuple(history.shape)}")
        images = prepare_vggt_history(history, self.config.wnm_geometry_resolution)
        images = images.to(next(self.adapter.parameters()).device)
        with torch.no_grad():
            result = self._aggregator(images)
        aggregated, patch_start = result if isinstance(result, tuple) else (result, getattr(self._aggregator, "patch_token_start", 0))
        aggregated = [x.detach() if x is not None else None for x in aggregated]
        target_t, target_h, target_w = self.config.wnm_geometry_target_grid
        tokens = self.adapter(aggregated, patch_start, target_t, (target_h, target_w))
        return tokens.reshape(tokens.shape[0], -1, tokens.shape[-1]) if tokens.ndim == 5 else tokens
