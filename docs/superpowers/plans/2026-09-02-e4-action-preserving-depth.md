# E4 Action-Preserving Depth Distillation Implementation Plan

**Goal:** Add action-aligned, magnitude-capped depth gradients for the shared SceneProjector while preserving the E1 action path.

**Architecture:** Compute action and depth gradients on SceneProjector independently, remove depth components opposing action, cap the effective depth/action norm ratio, then overwrite only SceneProjector gradients. Clip action/shared and depth-only parameters separately.

**Tech Stack:** PyTorch, Accelerate, torch.distributed, pytest.

## Global Constraints

- VGGT remains frozen and inference-only.
- E1 register concat and inference behavior remain unchanged.
- Initial E4 debug uses `lambda=0.05`, ratio cap `0.15`, gradient accumulation `1`, AMP disabled.
- E4 must not use global gradient clipping across depth-only parameters.

### Task 1: Gradient surgery helpers

**Files:** Modify `src/lerobot/policies/smolvla/gradient_diagnostic.py`; test `tests/policies/smolvla/test_gradient_diagnostic.py`.

- Add a pure `action_aligned_depth_gradient(action, depth, depth_weight, ratio_cap, eps)` helper.
- Project only conflicting depth components and cap weighted depth norm.
- Return final gradient and metrics for projection/cap status.
- Add tests for conflict projection, non-conflict preservation, ratio cap, and zero action gradient.

### Task 2: E4 configuration and training integration

**Files:** Modify `src/lerobot/policies/smolvla/configuration_smolvla.py`, `src/lerobot/policies/smolvla/modeling_smolvla.py`, `src/lerobot/scripts/lerobot_train.py`.

- Add E4 enable flag and ratio-cap configuration.
- Expose action/depth losses to the training loop when E4 is enabled.
- Compute DDP-averaged SceneProjector action/depth gradients, apply surgery, and overwrite SceneProjector `.grad` after backward.
- Clip SceneProjector and action parameters together, and DepthProbe separately.
- Keep E4 disabled by default and preserve existing diagnostic behavior.

### Task 3: Verification and documentation

- Run unit tests and a 100-step E4 debug command.
- Verify no NaN/DDP hang, frozen VGGT, projection/cap metrics, and effective ratio <= 0.15.
- Document E4 configuration and debug/30K commands in `docs/experiments_e0_e1_e2.md`.
