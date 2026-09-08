# WNM Geometry Adapter for SmolVLA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Condition SmolVLA action flow matching on 32 task-trained geometry tokens derived from four LIBERO RGB history frames by a frozen VGGT-Omega and the WNM-3D Geometry Adapter.

**Architecture:** Add a dedicated conditioner that loads the frozen VGGT aggregator from external WNM code and checkpoint paths while registering only the trainable adapter in the policy state dict. SmolVLA continues to encode the latest images normally, injects geometry tokens between image and language prefix tokens, and uses its unchanged action expert and flow-matching loss. Offline history comes from LeRobot delta indices; online history comes from a per-policy deque updated on every observation.

**Tech Stack:** Python 3.12, PyTorch, LeRobot, SmolVLA, VGGT-Omega, WNM-3D, pytest, PEFT.

## Global Constraints

- The new WNM geometry path and existing register-token path are mutually exclusive.
- VGGT-Omega is frozen, always in eval mode, and runs under `torch.no_grad()`.
- The WNM Geometry Adapter is randomly initialized, trainable, serialized, and restored.
- Use one configured LIBERO camera and exactly four history frames by default.
- Use a 512x512 VGGT input, a 32x32 source patch grid, and a 2x4x4 target grid by default.
- Do not modify SmolVLA action flow matching, suffix construction, Euler integration, or action loss.
- Do not add depth, geometry, static/dynamic, or language-selection losses.
- Do not serialize frozen VGGT-Omega parameters in policy checkpoints.
- Preserve existing E0/E1/E2/E4 behavior when the new switch is false.

---

### Task 1: Geometry Configuration and Observation History Contract

**Files:**
- Modify: `src/lerobot/policies/smolvla/configuration_smolvla.py`
- Test: `tests/policies/smolvla/test_wnm_geometry_adapter.py`

**Interfaces:**
- Produces: `SmolVLAConfig.use_wnm_geometry_tokens` and the `wnm_geometry_*` configuration fields.
- Produces: `SmolVLAConfig.observation_delta_indices -> list[int]` that remains `[0]` outside WNM mode and returns four historical indices in WNM mode.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_wnm_geometry_uses_history_delta_indices():
    cfg = make_config(use_wnm_geometry_tokens=True, n_obs_steps=4, wnm_geometry_history=4)
    assert cfg.observation_delta_indices == [-3, -2, -1, 0]


def test_existing_smolvla_keeps_single_observation():
    cfg = make_config(use_wnm_geometry_tokens=False, n_obs_steps=4)
    assert cfg.observation_delta_indices == [0]


def test_wnm_geometry_rejects_register_path_and_bad_history():
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_config(use_wnm_geometry_tokens=True, use_vggt_scene_tokens=True)
    with pytest.raises(ValueError, match="must equal n_obs_steps"):
        make_config(use_wnm_geometry_tokens=True, n_obs_steps=4, wnm_geometry_history=3)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `/home/jovyan/home/suziyang/.conda/envs/smolvla_libero/bin/python -m pytest tests/policies/smolvla/test_wnm_geometry_adapter.py -q`

Expected: FAIL because `use_wnm_geometry_tokens` and its validation do not exist.

- [ ] **Step 3: Add the minimal configuration fields and validation**

Add fields for enablement, history/image key, external WNM code/checkpoint paths, image resolution, patch size, encoder dtype/compute dtype, adapter dimension/heads/blocks, and target grid. Update `observation_delta_indices` only when the new path is enabled.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2. Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/lerobot/policies/smolvla/configuration_smolvla.py tests/policies/smolvla/test_wnm_geometry_adapter.py
git commit -m "feat: configure WNM geometry history"
```

### Task 2: Frozen VGGT and Trainable WNM Geometry Conditioner

**Files:**
- Create: `src/lerobot/policies/smolvla/wnm_geometry_conditioner.py`
- Modify: `tests/policies/smolvla/test_wnm_geometry_adapter.py`

**Interfaces:**
- Produces: `prepare_vggt_history(history: Tensor, resolution: int) -> Tensor`.
- Produces: `WNMGeometryConditioner(config: SmolVLAConfig, output_dim: int, aggregator: nn.Module | None = None, adapter: nn.Module | None = None)`.
- Produces: `WNMGeometryConditioner.forward(history: Tensor) -> Tensor` with shape `[B, target_t*target_h*target_w, output_dim]`.

- [ ] **Step 1: Write failing preprocessing and conditioner tests**

Use tiny injected fake modules that expose the same aggregator/adapter interfaces. Assert:

```python
history = torch.rand(2, 4, 3, 16, 20)
tokens = conditioner(history)
assert tokens.shape == (2, 32, 24)
assert fake_aggregator.seen_shape == (2, 4, 3, 32, 32)
assert not any(p.requires_grad for p in fake_aggregator.parameters())
assert all(p.requires_grad for p in fake_adapter.parameters())
assert not any(key.startswith("_aggregator") for key in conditioner.state_dict())
```

Backpropagate `tokens.square().mean()` and assert aggregator gradients are `None` while at least one adapter gradient is finite and nonzero.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `/home/jovyan/home/suziyang/.conda/envs/smolvla_libero/bin/python -m pytest tests/policies/smolvla/test_wnm_geometry_adapter.py -q`

Expected: FAIL because `wnm_geometry_conditioner` does not exist.

- [ ] **Step 3: Implement the conditioner**

Implement raw-range conversion, bicubic resize, shape checks, frozen external aggregator ownership through `self.__dict__["_aggregator"]`, lazy device/dtype placement, no-grad aggregation, cached-token detachment, and registered trainable adapter execution. Production construction dynamically imports `Aggregator` and `VGGTOmegaGeometryAdapter` after adding the configured WNM root to `sys.path`, and loads checkpoint keys using WNM's supported prefixes.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all Task 1-2 tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/lerobot/policies/smolvla/wnm_geometry_conditioner.py tests/policies/smolvla/test_wnm_geometry_adapter.py
git commit -m "feat: add WNM geometry conditioner"
```

### Task 3: Prefix and Action-Flow Integration

**Files:**
- Modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- Modify: `tests/policies/smolvla/test_wnm_geometry_adapter.py`

**Interfaces:**
- Consumes: `WNMGeometryConditioner.forward(history) -> geometry_tokens`.
- Produces: `VLAFlowMatching.encode_geometry(history: Tensor | None) -> Tensor | None`.
- Extends: `embed_prefix(..., geometry_tokens: Tensor | None = None)`.
- Extends: `forward(..., geometry_history: Tensor | None = None)` and `sample_actions(..., geometry_history: Tensor | None = None)`.

- [ ] **Step 1: Write failing prefix tests**

Construct a lightweight `VLAFlowMatching` fixture without loading pretrained models and assert geometry tokens occur after all image tokens but before language/state tokens. Assert their pad mask is true and their autoregressive mask value is zero. Also assert `encode_geometry(None)` returns `None` when disabled.

- [ ] **Step 2: Run and verify RED**

Run the focused pytest command. Expected: FAIL because the methods do not accept geometry inputs.

- [ ] **Step 3: Implement minimal model integration**

Instantiate `WNMGeometryConditioner` only under `use_wnm_geometry_tokens`. Compute geometry once per training forward or sampling prefill, cast its output to prefix device/dtype, and pass it into `embed_prefix`. Leave suffix embedding, masks, action projection, MSE loss, and Euler integration untouched.

- [ ] **Step 4: Run focused and existing SmolVLA tests**

Run:

```bash
/home/jovyan/home/suziyang/.conda/envs/smolvla_libero/bin/python -m pytest tests/policies/smolvla/test_wnm_geometry_adapter.py tests/policies/smolvla/test_vggt_depth_distillation.py tests/policies/smolvla/test_smolvla_rtc.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/lerobot/policies/smolvla/modeling_smolvla.py tests/policies/smolvla/test_wnm_geometry_adapter.py
git commit -m "feat: condition SmolVLA on geometry tokens"
```

### Task 4: Offline and Online Geometry History

**Files:**
- Modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- Modify: `tests/policies/smolvla/test_wnm_geometry_adapter.py`

**Interfaces:**
- Produces: `SmolVLAPolicy.prepare_wnm_geometry_history(batch, online: bool) -> Tensor | None`.
- Maintains: `_wnm_geometry_history: deque[Tensor]` initialized by `reset()`.

- [ ] **Step 1: Write failing history tests**

Assert offline `[B,4,C,H,W]` is returned unchanged from the configured image key. For online `[B,C,H,W]`, assert the sequence evolves as:

```text
[I0,I0,I0,I0]
[I0,I0,I0,I1]
[I0,I0,I1,I2]
[I0,I1,I2,I3]
```

Assert an absent key and an incorrect frame count raise clear `ValueError`s.

- [ ] **Step 2: Run and verify RED**

Run the focused pytest command. Expected: FAIL because online geometry history is not implemented.

- [ ] **Step 3: Implement history preparation and policy wiring**

Initialize the deque in `reset()`. Update it on every call to `select_action()` and `predict_action_chunk()` before deciding whether to generate a new chunk. Pass offline history directly during training. Avoid double append by performing exactly one history update per public policy inference call and passing the prepared tensor into `_get_action_chunk()`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the focused pytest command. Expected: all history tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add src/lerobot/policies/smolvla/modeling_smolvla.py tests/policies/smolvla/test_wnm_geometry_adapter.py
git commit -m "feat: maintain geometry observation history"
```

### Task 5: PEFT and Checkpoint Contract

**Files:**
- Modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- Modify: `tests/policies/smolvla/test_wnm_geometry_adapter.py`

**Interfaces:**
- Extends: `SmolVLAPolicy._get_default_peft_targets()` so the registered geometry adapter is a full saved/trainable module under PEFT.
- Preserves: non-strict base SmolVLA checkpoint loading with randomly initialized geometry adapter keys.

- [ ] **Step 1: Write failing PEFT/state-dict tests**

Assert the WNM mode adds the exact adapter module path to `modules_to_save`, legacy mode preserves the current empty list, conditioner state dict contains adapter keys, and contains no aggregator keys.

- [ ] **Step 2: Run and verify RED**

Run the focused pytest command. Expected: FAIL because the adapter is absent from PEFT modules-to-save.

- [ ] **Step 3: Implement PEFT configuration support**

Build `modules_to_save` conditionally without changing existing target-module regexes. Confirm `set_requires_grad()` explicitly keeps adapter parameters trainable in non-PEFT training.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the focused pytest command. Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add src/lerobot/policies/smolvla/modeling_smolvla.py tests/policies/smolvla/test_wnm_geometry_adapter.py
git commit -m "fix: preserve geometry adapter with PEFT"
```

### Task 6: Documentation and Verification

**Files:**
- Modify: `docs/experiments_e0_e1_e2.md`
- Modify: `tests/policies/smolvla/test_wnm_geometry_adapter.py` only if verification exposes a missing case.

**Interfaces:**
- Documents: the exact WNM geometry CLI fields and the single-camera LIBERO configuration.

- [ ] **Step 1: Add the WNM Geometry Adapter experiment section**

Document the frozen/trainable boundary, default shapes, mutually exclusive legacy switch, external checkpoint requirement, expected camera key `observation.images.image`, and a runnable `lerobot-train` configuration example.

- [ ] **Step 2: Run formatter/linter on changed Python files**

Run:

```bash
/home/jovyan/home/suziyang/.conda/envs/smolvla_libero/bin/python -m ruff check src/lerobot/policies/smolvla/configuration_smolvla.py src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/wnm_geometry_conditioner.py tests/policies/smolvla/test_wnm_geometry_adapter.py
/home/jovyan/home/suziyang/.conda/envs/smolvla_libero/bin/python -m ruff format --check src/lerobot/policies/smolvla/configuration_smolvla.py src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/wnm_geometry_conditioner.py tests/policies/smolvla/test_wnm_geometry_adapter.py
```

- [ ] **Step 3: Run final focused regression suite**

Run:

```bash
/home/jovyan/home/suziyang/.conda/envs/smolvla_libero/bin/python -m pytest tests/policies/smolvla/test_wnm_geometry_adapter.py tests/policies/smolvla/test_gradient_diagnostic.py tests/policies/smolvla/test_vggt_depth_distillation.py tests/policies/smolvla/test_smolvla_rtc.py -q
```

- [ ] **Step 4: Inspect final diff and state**

Run `git diff --check`, `git status --short`, and verify `nohup.out`, checkpoints, datasets, and training outputs are not staged.

- [ ] **Step 5: Commit documentation**

```bash
git add docs/experiments_e0_e1_e2.md
git commit -m "docs: describe WNM geometry training"
```
