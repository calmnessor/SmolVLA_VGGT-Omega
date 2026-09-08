# WNM Geometry Adapter for SmolVLA

## Goal

Add a WNM-3D patch-token geometry conditioning path to SmolVLA for LIBERO manipulation while preserving the existing register-token experiments for backward compatibility.

## Scope

The new path is the only target behavior for this implementation:

```text
LIBERO main-camera history [B,4,C,H,W]
  -> frozen VGGT-Omega patch features
  -> WNM Geometry Adapter
  -> 32 geometry tokens at SmolVLA hidden width
  -> SmolVLA prefix
  -> existing Action Expert and flow-matching loss
```

The existing `use_vggt_scene_tokens` register-token/depth-distillation path remains unchanged and is not combined with the new path.

## Architecture

### Geometry conditioner

Add a focused conditioner module under `lerobot/src/lerobot/policies/smolvla/` that owns a VGGT-Omega aggregator and a `VGGTOmegaGeometryAdapter` imported from WNM-3D. The aggregator is frozen, evaluated in `torch.no_grad()` with optional BF16 autocast, and its cached multi-level patch features are detached before entering the trainable adapter. The aggregator is kept external to the trainable checkpoint where practical; its checkpoint path is configuration-owned.

The adapter uses:

- `source_t=4`, `source_h=32`, `source_w=32`;
- `target_t=2`, `target_h=4`, `target_w=4`;
- four cached VGGT taps;
- adapter width 512 and two geometry blocks by default;
- output width equal to SmolVLA VLM text hidden size.

The conditioner accepts `[B,T,C,H,W]` RGB in `[0,1]` or uint8, resizes to the configured VGGT resolution, and returns `[B,32,D]`.

### SmolVLA integration

Add configuration fields with a dedicated `use_wnm_geometry_tokens` switch, history length, main image key, VGGT checkpoint/code paths, preprocessing resolution/dtype, adapter dimensions, and target grid. The existing register-token switch remains separate. Validate that geometry history equals `n_obs_steps` when enabled.

Extend `VLAFlowMatching.embed_prefix`, `forward`, and `sample_actions` with optional geometry inputs. Insert geometry tokens after image tokens and before language tokens. Geometry tokens use prefix attention mask value `0` and valid pad masks. The original action expert, suffix, Euler integration, and action loss remain unchanged.

### Dataset and online history

Set the new mode's observation deltas to `list(range(1 - n_obs_steps, 1))`, yielding `[-3,-2,-1,0]` for four frames. The model branch must read the raw `[0,1]` image tensor before SmolVLA's `[-1,1]` preprocessing. Online policy reset creates a bounded history deque; every environment observation is appended before action-queue decisions, and early episodes are left-padded with the first frame.

### Checkpoint and PEFT behavior

Existing SmolVLA checkpoints load non-strictly with the new adapter initialized from scratch. When PEFT is enabled, the geometry adapter is included in modules saved or otherwise remains explicitly trainable and serialized. Frozen VGGT weights are loaded from the configured external checkpoint and are not required to be duplicated in every policy checkpoint.

## Error handling

- Raise a clear error when geometry is enabled without a VGGT checkpoint/code path.
- Validate RGB shape `[B,T,3,H,W]`, frame count, and supported resolution/patch-size divisibility.
- Raise when the configured image key is absent or has incompatible shape.
- Reject simultaneous use of the old register-token path and the new WNM geometry path unless an explicit future composition is added.

## Tests

Add focused tests for:

1. Adapter/conditioner shape: `[2,4,3,512,512] -> [2,32,D]` using a lightweight fake aggregator or isolated adapter fixture.
2. Frozen VGGT and nonzero adapter gradient after a toy action loss.
3. Prefix token ordering and attention masks.
4. Four-frame observation delta indices and online history left-padding/update behavior.
5. Configuration validation and non-strict checkpoint compatibility.

GPU-dependent full VGGT execution remains an integration check; unit tests must not require the 1B checkpoint.

## Non-goals

- No changes to the WNM video DiT, VAE, navigation action head, or flow scheduler.
- No additional geometry/depth loss.
- No static/dynamic decomposition or language selector.
- No loading of the original 33-frame/450-token WNM adapter checkpoint.
