# E2 VGGT-Ω Depth Distillation Design

## Objective

Evaluate whether training-time depth distillation from frozen VGGT-Ω improves the E1 SmolVLA + VGGT-register policy, without changing the action inference architecture or deployment cost.

E2 is initialized from the E0 SmolVLA baseline checkpoint and trained for the same 30,000 steps as E1. The only intended experimental variable is the auxiliary depth objective.

## Model Graph

The action path is identical to E1:

```text
RGB views -> frozen VGGT registers -> SceneProjector -> prefix concat -> SmolVLA -> action loss
```

After `SceneProjector`, E2 adds a training-only branch:

```text
detached SmolVLA spatial image tokens -> Query
projected VGGT registers -> Key / Value
cross-attention -> spatial geometry tokens -> tiny per-patch DepthHead -> depth loss
```

The auxiliary branch must not be inserted before `SceneProjector` or modify the action prefix. Image tokens are detached so depth loss cannot update the frozen SmolVLA vision encoder. Projected registers remain attached so depth gradients update `SceneProjector`.

The image tokens must be the native, unpooled spatial patch tokens whose row and column positions can be reconstructed. Agent-view queries attend only to agent-view registers, and wrist-view queries attend only to wrist-view registers. Cross-view register mixing is outside the first E2 experiment.

The auxiliary attention and depth head are intentionally small and contain no dropout. Query and key may use linear projections, while the value path uses projected registers directly. The depth head is `Linear -> GELU -> Linear(1)`. Action forward is completed before auxiliary forward so the auxiliary branch does not perturb stochastic action operations through global RNG consumption.

## Teacher Outputs

The frozen VGGT-Ω model is constructed with `enable_depth=True` during training and returns `camera_and_register_tokens`, `depth`, and `depth_conf` in one forward pass. All VGGT parameters remain in evaluation mode with gradients disabled. Teacher depth is detached.

The inference path must not run VGGT depth or the auxiliary branch. It uses registers and the existing E1 action path only.

## Spatial Alignment

Teacher depth must undergo the same traceable geometric transform as the corresponding SmolVLA image. The implementation must not resize a teacher map independently unless the coordinate transform is proven equivalent. Teacher depth and confidence are first mapped into the SmolVLA image coordinate system, then pooled over each native SmolVLA patch grid cell.

For each image independently, retain pixels at or above that image's 20th confidence percentile. Within each student patch, average valid teacher depth pixels and compute the valid ratio. Patches with valid ratio below 25% are excluded. Agent and wrist camera losses are normalized separately and then averaged.

The final patch mask combines confidence validity, the 25% valid-ratio threshold, non-padding image regions, finite teacher depth/confidence, and positive teacher depth. Padding introduced by SmolVLA image preprocessing must never contribute to the loss.

## Loss

For valid student patches:

```text
L_depth = mean(|log(depth_student + eps) - log(depth_teacher + eps)|)
L_total = L_action + lambda_depth * L_depth
```

Student depth is constrained positive with `softplus(raw_depth) + 1e-6` before taking its logarithm. Teacher depth must be finite and greater than the same epsilon.

The first implementation does not include edge masking or gradient loss. `lambda_depth` is warmed up linearly from zero to `lambda_max` over the first 1,000 steps. A 100-500 step debug run records action/depth losses, projector gradient norms for each objective, and their cosine:

```text
ratio = ||lambda * grad_depth|| / ||grad_action||
cosine = dot(grad_action, grad_depth) / (||grad_action|| * ||grad_depth||)
```

The debug run records both the raw ratio `||grad_depth|| / ||grad_action||` and the full-weight ratio `||lambda_max * grad_depth|| / ||grad_action||`; it must not judge the final weight using the smaller warm-up value at the current debug step. The target full-weight ratio is approximately 0.1-0.2. Persistent cosine below -0.3 is evidence to reduce lambda. The initial candidate is `lambda_max=0.05`, subject to this diagnostic.

## Initialization and Fairness

E2 starts from the E0 30K baseline checkpoint. E1 and E2 use identical dataset, data order, batch size 16, seed 1000, optimizer, scheduler, and 30K step count. `SceneProjector` initialization must be identical between E1 and E2; adding auxiliary modules must not consume the RNG used for projector initialization. A dedicated initialization seed or saved projector initialization is acceptable.

## Required Verification

Before formal E2 training:

1. Verify `enable_depth=True` does not change VGGT register outputs relative to `enable_depth=False` on identical inputs, within floating-point tolerance.
2. With `lambda_depth=0`, compare old E1 and refactored E2 prefix shapes, token order, prefix values, action/noise outputs, and action loss; maximum prefix difference should be below `1e-6` where deterministic.
3. Verify teacher-to-student image transforms and patch pooling with synthetic coordinate-coded images/depth maps.
4. Verify image embeddings are unpooled spatial patch tokens with recoverable `(row, col)` ordering and distinct agent/wrist identity.
5. Verify padding patches, non-finite values, non-positive teacher depth, low confidence, and patches below the valid-ratio threshold are excluded.
6. Verify only `SceneProjector`, auxiliary attention, and `DepthHead` receive depth gradients; VGGT and SmolVLA vision parameters do not.
7. Verify the auxiliary branch has no dropout and runs after action forward without changing action-path RNG behavior.
8. Run the 100-500 step diagnostic and record raw ratio, full-`lambda_max` ratio, and cosine before selecting the formal lambda.

## Experiment and Evaluation

Formal E2 uses the E0 initialization and E1 protocol:

```text
dataset: lerobot/libero
steps: 30000
batch_size: 16
seed: 1000
two LIBERO camera views
```

Evaluation uses the same four suites, ten tasks per suite, and ten episodes per task. Depth computation is disabled at inference, so E2 and E1 have the same action-time graph and should be compared using the same evaluation protocol.

## Risks and Scope

VGGT depth is a teacher pseudo-label, not independent simulator ground truth. Therefore E2 is named depth distillation/consistency, not ground-truth depth supervision. If E2 does not improve E1, the result remains informative: register geometry may already be sufficient or dense metric depth may conflict with manipulation representations. MuJoCo ground-truth depth is deferred to a separate E3 experiment.
