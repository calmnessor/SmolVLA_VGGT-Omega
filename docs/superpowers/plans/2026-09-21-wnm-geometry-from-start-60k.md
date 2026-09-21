# WNM Geometry From-Start 60K Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch and verify a clean WNM Geometry 60K run with geometry active from the first optimization step.

**Architecture:** Reuse the existing WNM Geometry implementation without code changes. Load the original SmolVLA base weights, construct a newly initialized Geometry Adapter, freeze VGGT-Omega and the VLM backbone, and train the adapter and action modules with the standard action flow-matching loss.

**Tech Stack:** PyTorch, LeRobot SmolVLA, WNM-3D VGGT-Omega, LIBERO, CUDA.

## Global Constraints

- Use only GPU0 while the existing evaluation occupies GPU1.
- Use batch size 16 on the single training process for effective batch size 16.
- Use seed 1000, 60,000 steps, save frequency 10,000, and eight data workers.
- Use an independent output directory and log file.
- Do not change model source code or disturb the existing evaluation process.

---

### Task 1: Launch Validation

**Files:**
- Read: `checkpoints/smolvla_base/config.json`
- Read: `src/lerobot/policies/smolvla/wnm_geometry_conditioner.py`
- Create at runtime: `outputs/smolvla_libero_wnm_geometry_from_start_60k.log`

**Interfaces:**
- Consumes: original SmolVLA checkpoint, LIBERO dataset, WNM code, and VGGT-Omega checkpoint.
- Produces: one detached training process on GPU0 and its log.

- [ ] **Step 1: Confirm GPU0 is available and GPU1 evaluation remains alive**

Run: `nvidia-smi && ps -eo pid,args | rg lerobot_eval`

Expected: GPU0 has enough free memory; the existing evaluation process remains on GPU1.

- [ ] **Step 2: Confirm all input paths exist**

Run: `test -d checkpoints/smolvla_base && test -d datasets/libero && test -f /home/jovyan/home/suziyang/code/checkpoints/.VGGT-Omega/vggt_omega_1b_512.pt`

Expected: exit status 0.

- [ ] **Step 3: Launch the 60K run on GPU0**

Run the existing LeRobot training entry point with the values in the design document, `CUDA_VISIBLE_DEVICES=0`, and a detached log file.

Expected: one training process whose command includes `steps=60000`, `batch_size=16`, and the original `smolvla_base` path.

- [ ] **Step 4: Verify model construction and training startup**

Inspect the log for `matched=1283 missing=0 unexpected=0`, `Effective batch size: 16`, finite loss, finite gradient norm, and absence of traceback or CUDA OOM.

Expected: at least the first logged optimization interval completes successfully.

- [ ] **Step 5: Record monitoring commands**

Use `tail -f` for the experiment log and `nvidia-smi` for GPU utilization. Checkpoints are expected at 10K, 20K, 30K, 40K, 50K, and 60K.
