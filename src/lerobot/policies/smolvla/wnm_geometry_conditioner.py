from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

logger = logging.getLogger(__name__)


def prepare_vggt_history(history: Tensor, resolution: int) -> Tensor:
    if history.ndim != 5 or history.shape[2] != 3:
        raise ValueError(f"Expected history [B,T,3,H,W], got {tuple(history.shape)}")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if history.dtype == torch.uint8:
        x = history.float() / 255.0
    elif torch.is_floating_point(history):
        x = history.float()
    else:
        raise TypeError(f"Expected uint8 or floating-point history, got {history.dtype}")
    x = x.clamp(0.0, 1.0)
    b, t, c, h, w = x.shape
    x = F.interpolate(
        x.reshape(b * t, c, h, w), size=(resolution, resolution), mode="bicubic", align_corners=False
    )
    return x.reshape(b, t, c, resolution, resolution)


class WNMGeometryConditioner(nn.Module):
    def __init__(
        self, config, output_dim: int, aggregator: nn.Module | None = None, adapter: nn.Module | None = None
    ):
        super().__init__()
        self.config = config
        self.output_dim = output_dim
        self.encoder_dtype = getattr(
            torch, config.wnm_geometry_encoder_dtype.replace("float", "float"), torch.bfloat16
        )
        self.compute_dtype = getattr(
            torch, config.wnm_geometry_compute_dtype.replace("float", "float"), torch.bfloat16
        )
        if aggregator is None:
            if not config.wnm_geometry_code_path:
                raise ValueError("wnm_geometry_code_path is required for production WNM geometry")
            if not config.wnm_geometry_checkpoint:
                raise ValueError("wnm_geometry_checkpoint is required for production WNM geometry")
            root = str(Path(config.wnm_geometry_code_path).expanduser())
            if root not in sys.path:
                sys.path.insert(0, root)
            aggregator_cls = importlib.import_module("wnm_3d.modules.vggt_omega.models.aggregator").Aggregator
            aggregator = aggregator_cls(patch_size=config.wnm_geometry_patch_size)
            if config.wnm_geometry_checkpoint:
                state = torch.load(config.wnm_geometry_checkpoint, map_location="cpu", weights_only=True)
                state = state.get("state_dict", state) if isinstance(state, dict) else state
                state = state.get("model", state) if isinstance(state, dict) else state
                own_state = aggregator.state_dict()
                matched = {}
                for key, value in state.items():
                    candidates = [key]
                    for prefix in ("module.aggregator.", "aggregator.", "module."):
                        if key.startswith(prefix):
                            candidates.append(key[len(prefix) :])
                    for candidate in candidates:
                        if candidate in own_state:
                            matched[candidate] = value
                            break
                if not matched:
                    raise RuntimeError("No VGGT aggregator parameters matched checkpoint")
                missing, unexpected = aggregator.load_state_dict(matched, strict=False)
                log_fn = logger.warning if missing or unexpected else logger.info
                log_fn(
                    "VGGT checkpoint loaded: matched=%d missing=%d unexpected=%d",
                    len(matched),
                    len(missing),
                    len(unexpected),
                )
                if missing:
                    logger.warning("VGGT checkpoint missing sample: %s", missing[:10])
        self.__dict__["_aggregator"] = aggregator
        self._freeze_aggregator()
        if adapter is None:
            adapter_cls = importlib.import_module(
                "wnm_3d.modules.vggt_geometry_adapter"
            ).VGGTOmegaGeometryAdapter
            tt, hh, ww = config.wnm_geometry_target_grid
            adapter = adapter_cls(
                output_dim=output_dim,
                adapter_dim=config.wnm_geometry_adapter_dim,
                num_heads=config.wnm_geometry_adapter_heads,
                num_blocks=config.wnm_geometry_adapter_blocks,
                source_t=config.wnm_geometry_history,
                source_h=config.wnm_geometry_resolution // config.wnm_geometry_patch_size,
                source_w=config.wnm_geometry_resolution // config.wnm_geometry_patch_size,
                target_t=tt,
                target_h=hh,
                target_w=ww,
            )
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
            raise ValueError(
                f"Expected {self.config.wnm_geometry_history} history frames, got {tuple(history.shape)}"
            )
        images = prepare_vggt_history(history, self.config.wnm_geometry_resolution)
        device = next(self.adapter.parameters()).device
        self._aggregator.to(device=device)
        images = images.to(device=device)
        device_type = device.type
        with (
            torch.no_grad(),
            torch.autocast(device_type=device_type, dtype=self.encoder_dtype, enabled=device_type == "cuda"),
        ):
            result = self._aggregator(images)
        aggregated, patch_start = (
            result
            if isinstance(result, tuple)
            else (result, getattr(self._aggregator, "patch_token_start", 0))
        )
        aggregated = [x.detach() if x is not None else None for x in aggregated]
        target_t, target_h, target_w = self.config.wnm_geometry_target_grid
        with torch.autocast(device_type=device.type, dtype=self.compute_dtype, enabled=device.type == "cuda"):
            tokens = self.adapter(aggregated, patch_start, target_t, (target_h, target_w))
        if tokens.ndim == 5:
            tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        expected_tokens = target_t * target_h * target_w
        if tokens.ndim != 3 or tokens.shape[1] != expected_tokens or tokens.shape[2] != self.output_dim:
            raise ValueError(
                f"Expected geometry tokens [B,{expected_tokens},{self.output_dim}], got {tuple(tokens.shape)}"
            )
        if not torch.isfinite(tokens).all():
            raise FloatingPointError("WNM geometry adapter produced NaN or Inf")
        return tokens
