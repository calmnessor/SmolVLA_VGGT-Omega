# Gradient Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in 1000-step E2 gradient diagnostic that measures raw action/depth gradient alignment on SceneProjector without changing optimizer updates.

**Architecture:** Add pure diagnostic helpers for aligned flattening, DDP averaging, metric calculation, and summary aggregation. Extend the policy forward path with an opt-in component-loss return, then let the training loop compute diagnostics before the normal `accelerator.backward(loss_total)`. Rank 0 writes JSONL and summary files; all ranks participate in gradient all-reduce at identical diagnostic steps.

**Tech Stack:** PyTorch autograd, torch.distributed, LeRobot dataclass configuration, pytest.

## Global Constraints

- Diagnostic is disabled by default and must not change E0/E1/E2 behavior when disabled.
- Raw gradients define cosine and raw ratio; current lambda and lambda-max ratios are separate metrics.
- None gradients are replaced by zero tensors in parameter order.
- DDP diagnostic gradients are all-reduced and averaged before metrics are computed.
- Zero-norm measurements have `cosine=null` and `conflict=null`, and are excluded from cosine/conflict aggregates.
- First implementation is global-only; no suite/task grouping and no PCGrad.
- Diagnostic starts from E0 checkpoint via a normal E2 run and is limited to 1000 steps, sampling every 10 steps.

### Task 1: Add Diagnostic Math Helpers

**Files:**
- Create: `src/lerobot/policies/smolvla/gradient_diagnostic.py`
- Test: `tests/policies/smolvla/test_gradient_diagnostic.py`

**Interfaces:**
- `flatten_grads(grads, params) -> Tensor`
- `compute_gradient_metrics(action_grads, depth_grads, params, current_lambda, lambda_max, eps=1e-8) -> dict`
- `GradientMetricAccumulator.add(record)` and `.summary() -> dict`

- [ ] **Step 1: Write failing tests** for parameter-aligned None gradients, raw cosine versus weighted ratio, zero-norm null cosine, and summary percentiles.
- [ ] **Step 2: Run** `pytest tests/policies/smolvla/test_gradient_diagnostic.py -q` and confirm failure because the helper module is missing.
- [ ] **Step 3: Implement** aligned flattening, cosine/norm/ratios, and accumulator with `mean`, `median`, `p10`, `p95`, valid count, and conflict rate.
- [ ] **Step 4: Run** the focused tests and confirm they pass.
- [ ] **Step 5: Commit** `feat: add gradient diagnostic metric helpers`.

### Task 2: Add Opt-In Configuration and Component Loss Access

**Files:**
- Modify: `src/lerobot/policies/smolvla/configuration_smolvla.py`
- Modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`
- Test: `tests/policies/smolvla/test_vggt_depth_distillation.py`

**Interfaces:**
- Config fields: `enable_gradient_diagnostic=False`, `gradient_diagnostic_steps=1000`, `gradient_diagnostic_interval=10`, `gradient_diagnostic_log_interval=100`, `gradient_diagnostic_output_dir=None`.
- Policy method: `forward_with_components(batch) -> (loss_action, depth_loss, depth_lambda, output_dict)`.

- [ ] **Step 1: Write failing tests** asserting defaults are disabled and component losses preserve the existing total-loss value.
- [ ] **Step 2: Run** the focused tests and confirm failure.
- [ ] **Step 3: Implement** config fields and refactor the existing forward computation so the new method exposes unweighted action/depth losses while `forward()` retains its current return contract.
- [ ] **Step 4: Run** focused tests and confirm pass, including dtype and depth-disabled cases.
- [ ] **Step 5: Commit** `feat: expose E2 component losses for diagnostics`.

### Task 3: Integrate Diagnostic into Single- and Multi-GPU Training

**Files:**
- Modify: `src/lerobot/scripts/lerobot_train.py`
- Modify: `src/lerobot/configs/train.py` only if parser wiring requires it
- Test: `tests/scripts/test_gradient_diagnostic_training.py`

**Interfaces:**
- `run_gradient_diagnostic(policy, batch, accelerator, current_step, output_writer) -> dict | None`
- Rank 0 writes `<output_dir>/gradient_metrics.jsonl` and `<output_dir>/summary.json`.

- [ ] **Step 1: Write failing tests** for interval gating, normal backward remaining unchanged, rank-0-only writes, and all-reduced gradients.
- [ ] **Step 2: Run** focused tests and confirm failure.
- [ ] **Step 3: Implement** pre-backward `torch.autograd.grad(... retain_graph=True, create_graph=False, allow_unused=True)` for raw action and raw depth losses on unwrapped SceneProjector parameters; manually all-reduce flattened vectors when distributed; compute current and lambda-max ratios; do not assign diagnostic gradients to `.grad`.
- [ ] **Step 4: Integrate** the call into `update_policy` before `accelerator.backward(loss_total)`, limited by configured steps and interval, while preserving the existing optimizer path.
- [ ] **Step 5: Add** rank-0 JSONL writing and final summary writing at training shutdown, including `n_measurements`, `n_valid_gradient_measurements`, cosine statistics, ratios, mean depth loss, and valid depth patch ratio when available.
- [ ] **Step 6: Run** focused tests and a 2-process CPU/gloo integration test (or skip only the GPU-specific path with an explicit reason).
- [ ] **Step 7: Commit** `feat: integrate opt-in E2 gradient diagnostics`.

### Task 4: Add Documentation and 1000-Step Run Instructions

**Files:**
- Modify: `docs/experiments_e0_e1_e2.md`
- Create: `docs/gradient_diagnostic.md`

- [ ] **Step 1: Document** metric definitions, zero-gradient handling, DDP semantics, output schema, and interpretation thresholds as guidance rather than hard rules.
- [ ] **Step 2: Document** the exact E0-initialized E2 1000-step command, including `--policy.enable_gradient_diagnostic=true`, interval 10, and output directory.
- [ ] **Step 3: Run** Markdown/path checks and verify all referenced files and flags exist.
- [ ] **Step 4: Commit** `docs: document E2 gradient diagnostic`.

## Verification Checklist

- [ ] Default E0/E1/E2 forward and optimizer behavior is unchanged when diagnostic is disabled.
- [ ] Raw cosine never uses lambda-scaled depth gradients.
- [ ] `lambda_max_grad_ratio` remains informative during warmup.
- [ ] None gradients preserve parameter alignment.
- [ ] Zero-norm records use null cosine/conflict and are excluded from aggregate cosine/conflict statistics.
- [ ] In DDP, every rank enters diagnostic collectively and only rank 0 writes files.
- [ ] A 1000-step E2 run produces `gradient_metrics.jsonl` and `summary.json`.
