# SmolVLA + VGGT-Omega LIBERO Experiments

本文档记录当前项目的 E0、E1、E2 三组实验，包含模型结构、代码变更、训练参数、评估协议和正式结果。

## 1. 实验总览

| 实验 | 模型与训练目标 | 初始化 | 训练期额外目标 |
|---|---|---|---|
| E0 | SmolVLA baseline | SmolVLA base | action flow-matching loss |
| E1 | SmolVLA + VGGT-Omega register concat | E0 的 30K checkpoint | action loss |
| E2 | E1 + VGGT-Omega depth distillation | E0 的 30K checkpoint | action loss + depth distillation loss |

E1 和 E2 都从 E0 的最终 checkpoint 初始化，并重新训练 30,000 steps；它们不是从 E1 checkpoint 继续训练。

## 2. 共同训练设置

- Dataset: `lerobot/libero`，本地根目录 `datasets/libero`
- 观测：两路 RGB 图像（`image`、`image2`）、8D state
- 动作：7D action，chunk size 50
- Video backend: PyAV
- Steps: 30,000
- Seed: 1000
- DataLoader workers: 8
- Batch size 的有效值：16
- VLM: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`
- VLM layers: 16
- `freeze_vision_encoder=True`
- `train_expert_only=True`
- `train_state_proj=True`
- AdamW：learning rate `1e-4`，weight decay `1e-10`，betas `(0.9, 0.95)`
- Warmup: 1,000 steps
- Decay: 30,000 steps，最终 learning rate `2.5e-6`
- AMP: disabled

## 3. E0：SmolVLA baseline

### 方法

使用原始 SmolVLA 视觉编码器和 action policy，不使用 VGGT scene token，也不使用 depth loss。

### 代码状态

E0 使用项目原始 SmolVLA 实现。VGGT 相关功能均关闭：

```text
use_vggt_scene_tokens = false
use_vggt_depth_distillation = false
```

### 训练命令

```bash
cd ~/smolvla_libero/lerobot
CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --policy.path=../checkpoints/smolvla_base \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero \
  --dataset.root=../datasets/libero \
  --dataset.video_backend=pyav \
  --output_dir=../outputs/smolvla_libero_baseline \
  --job_name=smolvla_libero_baseline \
  --batch_size=16 \
  --steps=30000 \
  --seed=1000 \
  --policy.device=cuda \
  --num_workers=8 \
  --wandb.enable=false \
  --log_freq=100
```

Checkpoint:

```text
outputs/smolvla_libero_baseline/checkpoints/030000/pretrained_model
```

## 4. E1：VGGT-Omega register concat

### 方法

两路 RGB 图像分别 resize/crop 到 `512x512`，输入冻结的 VGGT-Omega 1B checkpoint。每个视角提取 16 个 register token：

```text
2 views x 16 registers = 32 tokens
VGGT output: [B, 32, 2048]
SceneProjector: Linear(2048 -> SmolVLA hidden size 960)
```

投影后的 `[B, 32, 960]` token 直接拼接到 SmolVLA prefix。VGGT 参数全部冻结，只有 projector 和原本可训练的 SmolVLA action 分支更新。

### 代码变更

主要提交：`0d799edb Add VGGT-Omega scene tokens to SmolVLA`。

关键实现位于：

- `src/lerobot/policies/smolvla/vggt_scene_encoder.py`：VGGT 动态加载、图像预处理、register 提取
- `src/lerobot/policies/smolvla/configuration_smolvla.py`：VGGT checkpoint、代码路径、分辨率和 register 数配置
- `src/lerobot/policies/smolvla/modeling_smolvla.py`：初始化 `FrozenVGGTSceneEncoder` 和 `scene_projector`，在 `embed_prefix` 中 concat scene tokens

E1 配置开关：

```text
use_vggt_scene_tokens = true
use_vggt_depth_distillation = false
vggt_image_resolution = 512
vggt_num_register_tokens = 16
```

### 训练参数

E1 正式 run 使用 `batch_size=16`、30,000 steps、seed 1000，其余参数与 E0 相同。初始化 checkpoint 为 E0 的 `030000` 模型。

Checkpoint:

```text
outputs/smolvla_libero_vggt_e1_run4/checkpoints/030000/pretrained_model
```

## 5. E2：VGGT depth distillation

### 方法

E2 保留 E1 的 register concat action 分支，并在训练期增加一个 depth probe：

```text
冻结 VGGT-Omega
  ├─ registers -> SceneProjector -> SmolVLA prefix -> action loss
  └─ depth + confidence -> teacher target

projected registers -> SpatialRegisterDepthProbe -> predicted depth
predicted depth 与 VGGT depth teacher 做 confidence-weighted log-L1 loss
```

VGGT teacher 使用 `no_grad`/detach，不参与反向传播。depth probe 只在训练模式启用，评估/推理时不改变 E1 的 action inference 路径。

总损失：

```text
L_total = L_action + lambda_depth * L_depth
```

实际配置：

```text
lambda_depth = 0.05
depth_distillation_warmup_steps = 1000
confidence_quantile = 0.2
min_valid_ratio = 0.25
```

### 代码变更

主要提交：

- `bea76806 feat: add E2 VGGT depth distillation training branch`
- `0cd26bec fix: run VGGT depth head in float32`
- `d7b6a899 perf: reuse VGGT teacher forward during E2 training`

新增/修改内容：

- `configuration_smolvla.py`：增加 depth distillation 开关和超参数
- `depth_distillation.py`：confidence mask、teacher depth pooling、log-L1 loss、warmup 权重
- `depth_probe.py`：轻量空间 register depth probe
- `vggt_scene_encoder.py`：同时返回 registers、depth 和 confidence，并冻结 VGGT
- `modeling_smolvla.py`：训练期计算 depth loss，并将其加入总 loss；单次 VGGT forward 复用 teacher 输出
- `test_vggt_depth_distillation.py`：depth 分支单元测试

### 两卡训练参数

E2 使用两卡 DDP：每卡 `batch_size=8`，有效 batch 为 `8 x 2 = 16`，因此与 E0/E1 的有效 batch 相同。Steps 仍为 30,000，seed 仍为 1000。

Checkpoint:

```text
outputs/smolvla_libero_vggt_e2_2gpu_v3/checkpoints/030000/pretrained_model
```

## 6. 评估协议

三组实验均使用 LIBERO 四个 suite：

```text
libero_spatial, libero_object, libero_goal, libero_10
```

每个 suite 10 个 task，每个 task 10 episodes，共 100 episodes/suite、400 episodes/实验。成功率为 episode success rate；总平均为四个 suite 成功率的宏平均。

E2 为加速评估，将 suite 分到两张 GPU：GPU0 评估 Spatial/Object，GPU1 评估 Goal/LIBERO-10，最后合并四个 suite 的结果。

## 7. 正式结果

| Method | Spatial | Object | Goal | LIBERO-10 | Macro Avg |
|---|---:|---:|---:|---:|---:|
| E0 SmolVLA baseline | 52.0% | 53.0% | 61.0% | 34.0% | **50.0%** |
| E1 + VGGT registers | 59.0% | 67.0% | 76.0% | 36.0% | **59.5%** |
| E2 + depth distillation | 52.0% | 59.0% | 70.0% | 43.0% | **56.0%** |
| E2 + depth distillation (`lambda=0.01`) | 51.0% | 66.0% | 65.0% | 46.0% | **57.0%** |

变化：

| Comparison | Spatial | Object | Goal | LIBERO-10 | Macro Avg |
|---|---:|---:|---:|---:|---:|
| E1 - E0 | +7.0 | +14.0 | +15.0 | +2.0 | **+9.5** |
| E2 - E0 | 0.0 | +6.0 | +9.0 | +9.0 | **+6.0** |
| E2 - E1 | -7.0 | -8.0 | -6.0 | +7.0 | **-3.5** |
| E2 (`lambda=0.01`) - E1 | -8.0 | -1.0 | -11.0 | +10.0 | **-2.5** |

结果文件：

```text
E0: LIBERO/outputs/eval_baseline/eval_info.json
E1: LIBERO/outputs/eval_vggt_e1_gpu0/eval_info.json
E2: outputs/eval_e2_gpu0/eval_info.json
    outputs/eval_e2_gpu1/eval_info.json
E2 (`lambda=0.01`):
    outputs/eval_e2_lambda001_gpu0/eval_info.json
    outputs/eval_e2_lambda001_gpu1_historical/eval_info.json
```

## 8. 结论

E1 是当前最佳整体结果：VGGT-Omega register token concat 将宏平均成功率从 50.0% 提升到 59.5%。E2 相比 E0 仍有 6.0 个百分点提升，但在当前 `lambda_depth=0.05` 和 confidence filtering 配置下，未超过 E1，反而比 E1 低 3.5 个百分点。E2 在 LIBERO-10 上从 36.0% 提升到 43.0%，但 Spatial、Object、Goal 分别下降 7、8、6 个百分点，说明当前 depth auxiliary objective 可能与主要 action objective 存在梯度冲突，或 VGGT pseudo-depth 对这些任务的监督并不完全匹配。

E2 评估仍使用与 E1 相同的推理结构；depth probe 仅用于训练监督，评估时不会额外输入 depth，也不会改变 action policy 的输入协议。




## 9. E2 梯度诊断（lambda=0.05）

为解释 E2 相比 E1 下降的原因，使用 E0 的 30K checkpoint 初始化，按两卡 DDP 运行 1,000 steps，每 10 steps 记录一次 SceneProjector 上的 action/depth 梯度。VGGT 保持冻结，诊断梯度通过 `autograd.grad` 获取并在两卡间手动 all-reduce。结果文件：

```text
outputs/e2_gradient_diagnostic_v5/gradient_metrics.jsonl
outputs/e2_gradient_diagnostic_v5/summary.json
```

共获得 100/100 个有效测量：

| Metric | Value |
|---|---:|
| Mean cosine | 0.0048 |
| Median cosine | 0.0045 |
| P10 cosine | -0.0246 |
| Conflict rate | 41.0% |
| Mean raw depth/action ratio | 23.59 |
| Mean weighted ratio | 0.580 |
| P95 weighted ratio | 1.389 |
| Mean ratio at lambda-max=0.05 | 1.179 |
| P95 ratio at lambda-max | 3.096 |

结果表明 depth 与 action 梯度整体接近正交，约 41% 的 batch 出现负余弦冲突；在 `lambda=0.05` 完全打开时，depth 梯度平均约为 action 梯度的 1.18 倍，P95 达到 3.10 倍。因此 E2 下降更可能来自 depth objective 的梯度竞争和权重过强。

## 10. E2 低权重消融（lambda=0.01）

基于上述诊断，启动与 E1/E2 相同的数据、seed、有效 batch 和 30K 总步数的低权重消融：

```text
depth_distillation_lambda = 0.01
per-GPU batch size = 8
GPU = 0,1
effective batch size = 16
seed = 1000
```

实验目录：

```text
outputs/smolvla_libero_vggt_e2_lambda001
```

该实验第一次运行在约 21K/30K 步收到外部 `SIGTERM`，只保存 `010000` 和 `020000` checkpoint，未生成 `030000`。随后修正断点命令，移除会覆盖 resume 解析的 `--policy.path`，从 `020000` 正确恢复，续训目标为 30,000 steps，剩余 10,000 steps。首次运行约 20K 步时 action loss 约 0.35、depth loss 约 0.068、total loss 约 0.33--0.35，曲线平稳下降。续训完成后生成 `030000` checkpoint，并完成四 suite 评估，结果见第 7 节和下方评估明细。

## 11. 当前结论与记录规则

- 当前最佳已完成模型仍为 E1，宏平均成功率 59.5%。
- E2 (`lambda=0.05`) 是已完成的 depth distillation 负结果，宏平均 56.0%。
- `lambda=0.01` 已生成 `030000` checkpoint 并完成四 suite 评估，正式结果为 51/66/65/46，宏平均 57.0%。
- 当前最佳已完成模型仍为 E1（59.5%）；`lambda=0.01` 相比 `lambda=0.05`（56.0%）提高 1.0 个百分点，但仍比 E1 低 2.5 个百分点。
- 后续报告区分训练完成、checkpoint 完整和评估完成。
