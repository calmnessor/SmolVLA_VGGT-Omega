# WNM Geometry Repeat-Current 60K Design

## Goal

Measure whether WNM Geometry's gain comes from genuine temporal history rather
than four image slots or the added VGGT/Adapter compute.

## Controlled Comparison

The treatment is identical to the completed WNM Geometry from-start 60K run,
except for one setting:

- Real-history run: `[I(t-3), I(t-2), I(t-1), I(t)]`
- Repeat-current run: `[I(t), I(t), I(t), I(t)]`

Set `policy.wnm_geometry_history_ablation=repeat_current`. Keep the original
`smolvla_base` initialization, 60,000 optimization steps, two GPUs, effective
batch size 16, seed 1000, optimizer and scheduler, frozen VGGT-Omega, trainable
Geometry Adapter, 32 output geometry tokens, dataset, and checkpoint cadence
unchanged.

Use a new output directory and do not resume from or overwrite any previous
experiment.

## Verification

Accept the launch only after the log confirms:

- `pretrained_path` points to `checkpoints/smolvla_base` and `resume` is false.
- `wnm_geometry_history_ablation` is `repeat_current`.
- The effective batch size is `8 x 2 = 16` and training is 60,000 steps.
- VGGT loading reports `matched=1283 missing=0 unexpected=0`.
- Initial loss and gradient norm are finite, and both ranks run without OOM.

After training, evaluate the same 400 LIBERO episodes used for the real-history
and baseline controls. A real-history advantage over repeat-current supports the
claim that temporal observations, rather than duplicated visual input, account
for part of the improvement.
