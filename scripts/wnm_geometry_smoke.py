"""Run a real VGGT-Omega -> WNM geometry adapter gradient smoke test."""

from __future__ import annotations

import argparse

import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.wnm_geometry_conditioner import WNMGeometryConditioner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--code-path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = SmolVLAConfig(
        use_wnm_geometry_tokens=True,
        n_obs_steps=4,
        wnm_geometry_history=4,
        wnm_geometry_code_path=args.code_path,
        wnm_geometry_checkpoint=args.checkpoint,
        wnm_geometry_target_grid=(2, 4, 4),
    )
    conditioner = WNMGeometryConditioner(config, output_dim=960).to(device)
    history = torch.rand(1, 4, 3, 512, 512, device=device)
    tokens = conditioner(history)
    if tokens.shape != (1, 32, 960):
        raise AssertionError(f"Unexpected geometry token shape: {tuple(tokens.shape)}")
    if not torch.isfinite(tokens).all():
        raise AssertionError("Geometry tokens contain NaN or Inf")
    tokens.square().mean().backward()
    if any(param.grad is not None for param in conditioner._aggregator.parameters()):
        raise AssertionError("Frozen Aggregator received gradients")
    adapter_grads = [param.grad for param in conditioner.adapter.parameters() if param.grad is not None]
    if not adapter_grads or not any(
        torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in adapter_grads
    ):
        raise AssertionError("Adapter did not receive finite non-zero gradients")
    print(f"real geometry smoke ok: shape={tuple(tokens.shape)} max_abs={tokens.abs().max().item():.6g}")


if __name__ == "__main__":
    main()
