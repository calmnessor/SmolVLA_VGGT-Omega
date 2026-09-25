# WNM Geometry Repeat-Current 60K Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch and verify the controlled two-GPU repeat-current 60K ablation.

**Architecture:** Reuse the exact completed WNM Geometry from-start 60K command. Change only the history ablation from `real` to `repeat_current` and use a new experiment name, output directory, log, and tmux session.

**Tech Stack:** LeRobot, PyTorch Accelerate, two NVIDIA A100 GPUs, tmux.

## Global Constraints

- Initialize from `/home/jovyan/home/suziyang/smolvla_libero/checkpoints/smolvla_base` with no resume.
- Run 60,000 steps with per-process batch size 8 on two GPUs.
- Save every 10,000 steps and never overwrite an existing output directory.
- Keep seed 1000 and every model, dataset, optimizer, and scheduler option equal to the real-history run.

---

### Task 1: Launch And Verify Training

**Files:**
- Create at runtime: `/home/jovyan/home/suziyang/smolvla_libero/outputs/smolvla_libero_wnm_geometry_repeat_current_from_start_60k_2gpu/`
- Create at runtime: `/home/jovyan/home/suziyang/smolvla_libero/outputs/smolvla_libero_wnm_geometry_repeat_current_from_start_60k_2gpu.log`

**Interfaces:**
- Consumes: the existing `repeat_current` policy configuration and `smolvla_base` checkpoint.
- Produces: a detached two-GPU training process and checkpoints every 10,000 steps.

- [ ] **Step 1: Confirm both GPUs and the output target are free**

Run `nvidia-smi`, inspect training processes, and require that the output directory and tmux session do not exist.

- [ ] **Step 2: Launch the matched experiment**

Run the completed real-history command with these three substitutions only:

```text
wnm_geometry_history_ablation=repeat_current
job/output name=smolvla_libero_wnm_geometry_repeat_current_from_start_60k_2gpu
tmux session=wnm_geometry_repeat_current_60k_2gpu
```

- [ ] **Step 3: Verify configuration and model loading**

Require the log to show `repeat_current`, `steps=60000`, `resume=False`, the base checkpoint path, effective batch size `8 x 2 = 16`, and VGGT `matched=1283 missing=0 unexpected=0`.

- [ ] **Step 4: Verify live optimization**

Wait for the first logged training update. Require finite loss and gradient norm, two active ranks, GPU memory on both devices, and no traceback, OOM, or distributed worker failure.
