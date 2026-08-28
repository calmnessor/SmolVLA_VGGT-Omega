# E2 VGGT-Ω Depth Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a training-only confidence-masked VGGT-Ω depth distillation branch to SmolVLA while preserving the E1 action and inference graph exactly.

**Architecture:** Extend the frozen VGGT wrapper to return registers plus depth/confidence during training. After the existing SceneProjector, keep the E1 register prefix unchanged and branch to a small per-view spatial-query/register-value cross-attention probe whose positive depth prediction is trained against VGGT pseudo-depth. Disable the probe and VGGT depth head during inference.

**Tech Stack:** PyTorch, LeRobot SmolVLA, VGGT-Ω local Python package, pytest, LIBERO.

## Global Constraints

- E2 starts from the E0 30K checkpoint and trains for 30,000 steps with batch size 16, seed 1000, and the E1 dataset/evaluation protocol.
- E1 action prefix token order, attention masks, and inference computation must remain unchanged.
- SmolVLA spatial image tokens are detached Query tensors; projected per-view VGGT registers are Key/Value tensors with Value taken directly from the projected registers.
- VGGT is always frozen and teacher tensors are detached; SmolVLA vision parameters never receive depth gradients.
- Agent-view queries attend only to agent-view registers; wrist-view queries attend only to wrist-view registers.
- The auxiliary attention and depth head contain no dropout and run after action forward.
- Teacher patch masks require top-80-percent per-image confidence, finite and positive depth, non-padding pixels, and at least 25-percent valid pixels per student patch.
- Student depth is `softplus(raw_depth) + 1e-6`; loss is confidence-masked per-camera log-L1 averaged across cameras.
- `lambda_depth` warms up linearly from 0 to `lambda_max` over 1,000 steps; select `lambda_max` using raw and full-weight projector gradient ratios and gradient cosine from a 100-500 step debug run.

### Task 1: Lock down current interfaces and write regression tests

**Files:**
- Create: `tests/policies/smolvla/test_vggt_depth_distillation.py`
- Inspect/modify: `src/lerobot/policies/smolvla/vggt_scene_encoder.py`
- Inspect/modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`

**Interfaces:**
- Tests establish expected signatures for VGGT output, patch pooling, lambda schedule, and gradient routing before implementation.

- [ ] **Step 1: Write failing unit tests** for: per-image confidence percentile masking; finite/positive depth filtering; 25-percent patch validity; per-camera averaging; softplus positivity; lambda warm-up; and per-view register selection.**
- [ ] **Step 2: Run `pytest tests/policies/smolvla/test_vggt_depth_distillation.py -q` and verify the new tests fail for missing interfaces.**
- [ ] **Step 3: Add deterministic synthetic fixtures** using coordinate-coded depth/confidence arrays and a fake `[B, 2, R, D]` register tensor so tests do not load the 4.3GB VGGT checkpoint.**
- [ ] **Step 4: Commit the test scaffolding** with `git add tests/policies/smolvla/test_vggt_depth_distillation.py && git commit -m "test: define E2 depth distillation contracts"`.**

### Task 2: Extend FrozenVGGTSceneEncoder with training teacher outputs

**Files:**
- Modify: `src/lerobot/policies/smolvla/vggt_scene_encoder.py`
- Test: `tests/policies/smolvla/test_vggt_depth_distillation.py`

**Interfaces:**
- Add `FrozenVGGTSceneEncoder.forward(images, include_depth=False) -> Tensor | dict[str, Tensor]`.
- Register-only mode returns `[B, 32, 2048]` exactly as E1.
- Teacher mode returns `{"registers": [B, 32, 2048], "depth": [B, 2, H, W, 1], "depth_conf": [B, 2, H, W]}` with detached tensors.

- [ ] **Step 1: Implement a fake-model unit test** asserting register-only output shape and teacher output keys without constructing the real checkpoint.**
- [ ] **Step 2: Implement `enable_depth` construction only when teacher mode is requested**, preserving the E1 lightweight register-only path for inference.**
- [ ] **Step 3: Keep VGGT loading via `self.__dict__["_model"]`, force `eval()`, and set all parameters `requires_grad=False`; run depth under `torch.no_grad()`.**
- [ ] **Step 4: Run the synthetic encoder tests and a real checkpoint smoke test** with two random 512 inputs, asserting register shape, depth/confidence finiteness, and no VGGT gradients.**
- [ ] **Step 5: Commit with `git add src/lerobot/policies/smolvla/vggt_scene_encoder.py tests/policies/smolvla/test_vggt_depth_distillation.py && git commit -m "feat: expose frozen VGGT depth teacher outputs"`.**

### Task 3: Implement traceable teacher-to-patch alignment and loss utilities

**Files:**
- Create or modify: `src/lerobot/policies/smolvla/depth_distillation.py`
- Test: `tests/policies/smolvla/test_vggt_depth_distillation.py`

**Interfaces:**
- `pool_teacher_to_patch_grid(depth, confidence, image_valid_mask, out_hw, confidence_quantile=0.2, min_valid_ratio=0.25) -> tuple[Tensor, Tensor]`.
- `confidence_masked_log_l1(student_depth, teacher_depth, valid_mask, eps=1e-6) -> Tensor`.
- `depth_lambda(step, lambda_max, warmup_steps=1000) -> float`.

- [ ] **Step 1: Add failing coordinate-coded pooling tests** that distinguish resize/padding coordinates, reject non-finite/non-positive depth, remove low-confidence pixels, and ignore patches under the valid-ratio threshold.**
- [ ] **Step 2: Implement pooling in the shared image coordinate system**: apply the explicit SmolVLA transform metadata, construct non-padding validity, then pool valid teacher pixels inside each native student patch rather than resizing a pre-pooled map.**
- [ ] **Step 3: Implement per-image 20th-percentile confidence filtering and per-camera loss normalization.**
- [ ] **Step 4: Implement stable log-L1 with `softplus` student depth handled by the caller and zero-valid-patch behavior returning a finite zero loss.**
- [ ] **Step 5: Run utility tests and commit with `git add src/lerobot/policies/smolvla/depth_distillation.py tests/policies/smolvla/test_vggt_depth_distillation.py && git commit -m "feat: add confidence-masked depth distillation utilities"`.**

### Task 4: Add the small Spatial-query/Register-value DepthProbe

**Files:**
- Create: `src/lerobot/policies/smolvla/depth_probe.py`
- Modify: `src/lerobot/policies/smolvla/configuration_smolvla.py`
- Test: `tests/policies/smolvla/test_vggt_depth_distillation.py`

**Interfaces:**
- `SpatialRegisterDepthProbe(image_dim, register_dim, query_dim=128) -> nn.Module`.
- `forward(image_tokens: Tensor, projected_registers: Tensor, patch_hw: tuple[int, int]) -> Tensor` returns raw per-patch depth `[B, H_p, W_p]`.

- [ ] **Step 1: Write shape and gradient tests** for two cameras, unequal patch grids, direct register Value path, and no dropout modules.**
- [ ] **Step 2: Implement per-view projections `q_proj(image_tokens.detach())`, `k_proj(registers)`, scaled dot-product attention, direct projected-register Values, and a tiny `Linear -> GELU -> Linear(1)` head.**
- [ ] **Step 3: Return only native spatial patch outputs and verify agent/wrist branches use their matching 16-register slices.**
- [ ] **Step 4: Add E2 config fields** for `use_vggt_depth_distillation`, `depth_distillation_lambda`, `depth_distillation_warmup_steps`, and mask thresholds with defaults preserving E1 behavior.**
- [ ] **Step 5: Run tests and commit with `git add src/lerobot/policies/smolvla/depth_probe.py src/lerobot/policies/smolvla/configuration_smolvla.py tests/policies/smolvla/test_vggt_depth_distillation.py && git commit -m "feat: add spatial register-value depth probe"`.**

### Task 5: Integrate E2 training branch without changing E1 action path

**Files:**
- Modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- Modify: `src/lerobot/policies/smolvla/vggt_scene_encoder.py`
- Test: `tests/policies/smolvla/test_vggt_depth_distillation.py`

**Interfaces:**
- Preserve `embed_prefix()` output shape/order for E1.
- Add an internal training-only path returning action loss plus `depth_loss` and diagnostics; inference calls register-only VGGT and never creates/executes the depth branch.

- [ ] **Step 1: Capture a deterministic E1 reference** for prefix tensor, attention masks, action/noise output, and action loss on one fixed batch.**
- [ ] **Step 2: Refactor prefix construction only enough to retain per-view native image patch tokens**, preserving the reference action prefix byte-for-byte within `1e-6`.**
- [ ] **Step 3: Execute action forward first, then request VGGT depth teacher and run the auxiliary probe with detached image queries and attached projected register K/V.**
- [ ] **Step 4: Compute two independently normalized camera losses, average them, apply warm-up lambda, and return `loss_action`, `loss_depth`, `loss_total`, valid-patch ratios, and gradient diagnostics.**
- [ ] **Step 5: Add explicit training/inference tests** proving E1 mode is unchanged, E2 inference has no depth call, and depth gradients reach only projector/probe/head.**
- [ ] **Step 6: Run `pytest tests/policies/smolvla/test_vggt_depth_distillation.py -q` plus the existing SmolVLA tests and commit with `git add src/lerobot/policies/smolvla/modeling_smolvla.py src/lerobot/policies/smolvla/vggt_scene_encoder.py tests/policies/smolvla/test_vggt_depth_distillation.py && git commit -m "feat: integrate training-only VGGT depth distillation"`.**

### Task 6: Add gradient diagnostics and a 100-step E2 debug runner

**Files:**
- Create: `scripts/debug_smolvla_vggt_depth.py`
- Modify if needed: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- Test: `tests/policies/smolvla/test_vggt_depth_distillation.py`

- [ ] **Step 1: Implement separate action-only and depth-only projector gradient extraction** using `torch.autograd.grad`, then compute raw ratio, full-`lambda_max` ratio, and cosine without changing optimizer gradients.**
- [ ] **Step 2: Run the debug script for 100 steps from the E0 checkpoint** with two synthetic or real LIBERO batches, logging action/depth losses, valid ratios, raw/full ratios, cosine, NaN checks, and trainable parameter names.**
- [ ] **Step 3: Verify the full-weight projector ratio is near 0.1-0.2 and reduce/increase `lambda_max` based on evidence; retain the chosen value in the experiment command/config.**
- [ ] **Step 4: Run the regression suite again and commit with `git add scripts/debug_smolvla_vggt_depth.py tests/policies/smolvla/test_vggt_depth_distillation.py && git commit -m "test: add E2 depth gradient diagnostics"`.**

### Task 7: Formal E2 training and matched evaluation

**Files:**
- Create: `scripts/run_smolvla_vggt_e2.sh`
- Create: `docs/experiments/e2-vggt-depth-distillation.md`

- [ ] **Step 1: Record the selected lambda, debug metrics, checkpoint hash, VGGT checkpoint path, and exact E0/E1 protocol before formal training.**
- [ ] **Step 2: Launch 30,000-step E2 from the E0 baseline into a fresh output directory using GPU selection and persistent tmux, with depth distillation enabled only for training.**
- [ ] **Step 3: Verify checkpoints contain SceneProjector, DepthProbe, and DepthHead weights but do not contain VGGT-Ω weights.**
- [ ] **Step 4: Evaluate E2 with the same four LIBERO suites, ten tasks per suite, ten episodes per task, and the same EGL/GPU protocol as E1.**
- [ ] **Step 5: Report Spatial/Object/Goal/Long/Avg beside E0 and E1, explicitly labeling E2 as VGGT pseudo-depth distillation and recording whether inference depth is disabled.**
- [ ] **Step 6: Commit run documentation and the launcher with `git add scripts/run_smolvla_vggt_e2.sh docs/experiments/e2-vggt-depth-distillation.md && git commit -m "docs: record E2 depth distillation experiment"`.**
