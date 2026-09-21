# WNM Geometry From-Start 60K Design

## Goal

Train WNM Geometry for 60,000 steps with geometry enabled from step 1, using the same original SmolVLA initialization that produced the E0 baseline.

## Experimental Contract

- Initialize SmolVLA from `/home/jovyan/home/suziyang/smolvla_libero/checkpoints/smolvla_base`.
- Do not initialize from the E0 `030000` checkpoint and do not resume an optimizer.
- Load the frozen VGGT-Omega checkpoint with all aggregator parameters matched.
- Train the Geometry Adapter, action expert, state projection, and action projections.
- Use four real consecutive main-camera frames and emit 32 geometry tokens of width 960.
- Train for 60,000 steps with one scheduler: 1,000-step warmup followed by cosine decay through step 60,000.
- Preserve effective batch size 16. While GPU1 evaluates the continued baseline, run on GPU0 with per-process batch size 16.
- Save every 10,000 steps in a new output directory. Do not modify or overwrite earlier WNM or baseline outputs.

## Verification

The launch is accepted only after the log confirms the original initialization path, `resume=false`, 60,000 steps, effective batch size 16, complete VGGT checkpoint loading, finite loss and gradient norm, and creation of the first training steps without OOM.

