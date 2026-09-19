import pytest
import torch
from torch import nn

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.wnm_geometry_conditioner import (
    apply_history_ablation,
    prepare_vggt_history,
)


def test_history_ablation_repeat_current_replaces_all_frames():
    history = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1, 1)
    result = apply_history_ablation(history, "repeat_current")
    assert result.flatten().tolist() == [3.0, 3.0, 3.0, 3.0]


def test_history_ablation_reverse_preserves_temporal_values():
    history = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1, 1)
    result = apply_history_ablation(history, "reverse")
    assert result.flatten().tolist() == [3.0, 2.0, 1.0, 0.0]


def test_history_ablation_rejects_unknown_mode():
    history = torch.zeros(1, 4, 1, 1, 1)
    with pytest.raises(ValueError, match="Unknown history ablation"):
        apply_history_ablation(history, "bad")



def test_prepare_vggt_history_scales_uint8_without_saturation():
    history = torch.tensor([[[[[0, 128], [255, 64]]]]], dtype=torch.uint8).expand(1, 4, 3, 2, 2)
    result = prepare_vggt_history(history, 2)
    assert result.dtype == torch.float32
    assert torch.isclose(result[0, 0, 0, 0, 1], torch.tensor(128 / 255), atol=1e-5)


def test_wnm_config_validates_shapes_and_history():
    cfg = SmolVLAConfig(use_wnm_geometry_tokens=True, n_obs_steps=4, wnm_geometry_history=4)
    assert cfg.observation_delta_indices == [-3, -2, -1, 0]
    with pytest.raises(ValueError, match="divisible"):
        SmolVLAConfig(
            use_wnm_geometry_tokens=True, n_obs_steps=4, wnm_geometry_history=4, wnm_geometry_resolution=513
        )
    with pytest.raises(ValueError, match="history ablation"):
        SmolVLAConfig(wnm_geometry_history_ablation="bad")


def test_conditioner_state_does_not_register_external_aggregator():
    from lerobot.policies.smolvla.wnm_geometry_conditioner import WNMGeometryConditioner

    class Agg(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

    class Adapter(nn.Module):
        def forward(self, *_args):
            return torch.zeros(1, 2, 4, 24)

    cfg = SmolVLAConfig(use_wnm_geometry_tokens=True, n_obs_steps=4, wnm_geometry_history=4)
    conditioner = WNMGeometryConditioner(cfg, 24, aggregator=Agg(), adapter=Adapter())
    assert not any(k.startswith("_aggregator") for k in conditioner.state_dict())


def test_conditioner_requires_checkpoint_for_internal_aggregator():
    from lerobot.policies.smolvla.wnm_geometry_conditioner import WNMGeometryConditioner

    cfg = SmolVLAConfig(
        use_wnm_geometry_tokens=True,
        n_obs_steps=4,
        wnm_geometry_history=4,
        wnm_geometry_code_path="/tmp/wnm-code",
        wnm_geometry_checkpoint=None,
    )
    with pytest.raises(ValueError, match="wnm_geometry_checkpoint"):
        WNMGeometryConditioner(cfg, 24)


def test_conditioner_reports_partial_checkpoint_load(monkeypatch, caplog, tmp_path):
    from lerobot.policies.smolvla import wnm_geometry_conditioner as module
    from lerobot.policies.smolvla.wnm_geometry_conditioner import WNMGeometryConditioner

    class Agg(nn.Module):
        def __init__(self, patch_size=None):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))

    class Adapter(nn.Module):
        def forward(self, *_args):
            return torch.zeros(1, 2, 4, 24)

    monkeypatch.setattr(module.importlib, "import_module", lambda _name: type("M", (), {"Aggregator": Agg})())
    monkeypatch.setattr(module.torch, "load", lambda *_args, **_kwargs: {"weight": torch.ones(1)})
    cfg = SmolVLAConfig(
        use_wnm_geometry_tokens=True,
        n_obs_steps=4,
        wnm_geometry_history=4,
        wnm_geometry_code_path=str(tmp_path),
        wnm_geometry_checkpoint=str(tmp_path / "checkpoint.pt"),
    )
    with caplog.at_level("WARNING"):
        WNMGeometryConditioner(cfg, 24, adapter=Adapter())
    assert "missing=1" in caplog.text
