# YOLO Distillation Methods Documentation

This document describes the implementation of knowledge distillation (KD) methods integrated into the MIT-licensed YOLO repository.

---

## Workflow

```
Step 1: Train teacher (v9-s) on target dataset  -> teacher has task knowledge
Step 2: Train student (v9-t) baseline           ->  no distillation, reference
Step 3: Train student (v9-t) with CWD           ->  guided by experienced teacher
Step 4: Train student (v9-t) with MGD           ->  guided by experienced teacher
```

The comparison script (`scripts/run_comparison.py`) follows this order automatically.

---

## Implementation Details

### Configuration (`task/train.yaml`)

```yaml
# Distillation settings
teacher_weight: None  # Path to teacher model weights (.pt or .ckpt)
teacher_model: None   # Teacher model architecture name (e.g. v9-s)

loss:
  distiller_type: cwd  # Options: cwd, mgd
  objective:
    BCELoss: 0.5
    BoxLoss: 7.5
    DFLoss: 1.5
```

- `teacher_weight`: Path to a trained teacher checkpoint. Accepts both `.pt` and Lightning `.ckpt` files.
- `teacher_model`: Architecture of the teacher (e.g., `v9-s`, `v9-m`).
- `distiller_type`: Selects the distillation strategy.

### Feature Hook Extraction (`yolo/tools/distill_loss.py`)

Intermediate feature maps are extracted from both teacher and student models via **forward hooks** registered on layers `6, 8, 12, 15, 18, 21` (backbone + neck). These correspond to the `conv2` submodule at each layer depth.

The teacher model is frozen (`eval()` mode, `requires_grad=False`). It runs a forward pass each batch only to populate the hooks. Hooks are registered at the start of each epoch and removed at the end.

---

### 1. Channel-Wise Distillation (CWD)

CWD aligns the per-channel spatial distributions between teacher and student. For each feature map it softmaxes spatial locations within each channel and minimises the KL divergence between teacher and student distributions.

**Data flow:**
```
Student feature (s_chan)  ->  align (1×1 Conv + BN, s_chan→t_chan)  ->  channel-softmax
Teacher feature (t_chan)  ->  channel-softmax
Loss = KL(teacher_dist || student_dist) per channel, averaged over all layers
```

**Hyperparameters:**
- `distill_weight = 0.3` — fixed scaling applied to the CWD loss
- `tau = 1.0` — softmax temperature

---

### 2. Masked Generative Distillation (MGD)

MGD randomly masks 65% of student feature map positions, then trains a small generation network to reconstruct the full teacher feature map from the masked student features. This forces the student to learn richer, more complete representations.

**Data flow:**
```
Student feature (s_chan)  ->  align (1×1 Conv, no BN, s_chan→t_chan)
                          ->  random spatial mask (N,1,H,W), 65% zeroed, broadcast over channels
                          ->  generation network (3×3 Conv + ReLU + 3×3 Conv, t_chan→t_chan)
                          ->  MSE vs raw teacher features (no normalization)
Loss = MSELoss(sum)(generated, teacher) / N * alpha_mgd
```

**Hyperparameters:**
- `alpha_mgd = 0.0002` — scales the MSE loss to be comparable to detection losses
- `lambda_mgd = 0.65` — fraction of spatial positions masked (65%)

**Cosine warmup schedule:** MGD uses a cosine-annealed distillation weight that ramps from `0.1` to `1.0` over all training epochs:

```python
distill_weight = ((1 - cos(epoch * π / total_epochs)) / 2) * (1 - 0.1) + 0.1
```

**Key design notes matching the original paper (arxiv:2205.01529):**
- Mask shape is `(N, 1, H, W)` — spatial mask broadcast across all channels (not per-channel)
- Alignment uses a pure `1×1 Conv2d`, **no BatchNorm** — BN would change scale/shift and interfere with the generation network learning
- MSE target is **raw teacher features**, not BN-normalized — the generation network learns to reconstruct the teacher's actual activations
- `MSELoss(reduction='sum') / N` — matches the paper exactly

---

### 3. Feature Imitation with Masking (Work in Progress)

Still under progress.

---

## Distillation Network Optimizer

The generation network (`MGDLoss.generation`) and channel alignment layers (`FeatureLoss.align_module`) are **not** part of the YOLO `self.model` object. They live inside `DistillationLoss → FeatureLoss`, a separate `nn.Module` hierarchy.

To ensure they are trained, `configure_optimizers` explicitly adds them as a second parameter group:

```python
def configure_optimizers(self):
    optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
    if self.loss_fn.use_distill:
        distill_params = list(self.loss_fn.distiller.distill_loss_fn.parameters())
        optimizer.add_param_group({'params': distill_params})
    scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
    return [optimizer], [scheduler]
```

In `yzd-v/MGD`, the loss modules are stored in an `nn.ModuleDict` alongside the student model, so they are automatically included when the optimizer is initialized with `model.parameters()`. 

The distillation networks share the same optimizer type, learning rate, and schedule as the detection backbone.

### Changes required for distillation runs

**1. LR warmup lambda count mismatch (`yolo/utils/model_utils.py`)**

`create_scheduler` originally hardcoded 3 `lr_lambdas` for `LambdaLR`, matching the 3 base param groups from `create_optimizer`. With distillation, `configure_optimizers` adds a 4th group, causing:
```
ValueError: Expected 4 lr_lambdas, but got 3
```
Fix: build the lambda list dynamically from the optimizer's actual param group count:
```python
lr_lambdas = [lambda2] + [lambda1] * (len(optimizer.param_groups) - 1)
warmup_schedule = LambdaLR(optimizer, lr_lambda=lr_lambdas)
```

**2. `max_lr` list too short for the distill param group (`yolo/tools/solver.py`)**

`create_optimizer` sets `optimizer.max_lr = [0.1, 0, 0]` for the 3 base groups. The custom `next_batch` method iterates all param groups and indexes into `max_lr`, causing an `IndexError` when the 4th distillation group is present.
Fix: extend `max_lr` when the distillation group is added:
```python
if self.loss_fn.use_distill:
    distill_params = list(self.loss_fn.distiller.distill_loss_fn.parameters())
    optimizer.add_param_group({'params': distill_params})
    optimizer.max_lr.append(0)
```

---

## Loss Integration (`yolo/tools/loss_functions.py`)

Distillation is combined with the standard YOLO detection loss inside `DualLoss`:

```
total_loss = BoxLoss + DFLoss + BCELoss + distill_weight * DistillLoss
```

The detection loss trains the student to detect objects. The distillation loss additionally trains the student's backbone/neck to produce feature representations similar to the teacher's.

---

## Training Process (`yolo/tools/solver.py`)

Each training step:
1. Student forward pass — hooks collect student feature maps at layers 6, 8, 12, 15, 18, 21.
2. Teacher forward pass (no grad) — hooks collect teacher feature maps.
3. Distillation loss computed from collected features (align + KL/MGD generation).
4. Total loss = detection loss + `distill_weight` × distillation loss.
5. Backprop updates: student backbone, alignment layers, and generation network together.

Hooks are registered at the start of each epoch and removed at the end.

---

## Validation Config (`task/validation.yaml`)

```yaml
nms:
  min_confidence: 0.25   # filters low-confidence boxes before mAP eval
  min_iou: 0.7
  max_bbox: 300
```

---

## Usage

### Step 1 — Train the teacher on your dataset

```shell
python yolo/lazy.py task=train \
    model=v9-s \
    dataset=football-players \
    name=v9s-teacher \
    use_wandb=True
```

### Step 2 — Train student with distillation using the trained teacher

Find the checkpoint saved under `runs/train/v9s-teacher/`:

```shell
# With CWD
python yolo/lazy.py task=train \
    model=v9-t \
    dataset=football-players \
    name=v9t-cwd \
    task.teacher_weight=runs/train/v9s-teacher/.../checkpoints/epoch=49.ckpt \
    task.teacher_model=v9-s \
    task.loss.distiller_type=cwd \
    use_wandb=True

# With MGD
python yolo/lazy.py task=train \
    model=v9-t \
    dataset=football-players \
    name=v9t-mgd \
    task.teacher_weight=runs/train/v9s-teacher/.../checkpoints/epoch=49.ckpt \
    task.teacher_model=v9-s \
    task.loss.distiller_type=mgd \
    use_wandb=True
```

> **Note:** Checkpoint filenames contain `=` characters (e.g. `epoch=49-step=1350.ckpt`). Wrap the path in single quotes when passing via command line so Hydra does not misparse it as a key=value override.

### Run full comparison automatically

The comparison script handles teacher training and checkpoint discovery automatically:

```shell
# Full run (trains teacher first, then student baseline/cwd/mgd)
python scripts/run_comparison.py --dataset football-players --epochs 50 --teacher-epochs 50

# Skip teacher training if already trained
python scripts/run_comparison.py \
    --skip-teacher \
    --teacher-weight runs/train/v9s-teacher/.../checkpoints/epoch=49.ckpt

# Quick smoke test
python scripts/run_comparison.py --dataset mock --epochs 5 --teacher-epochs 5
```

---

## Code Changes from Original YOLO

This section documents every change made to the original YOLO repository to add distillation support.

### New Files Added

| File | Description |
|------|-------------|
| `yolo/tools/distill_loss.py` | Entire distillation loss module — see classes below |

**`yolo/tools/distill_loss.py`** contains four classes:
- `CWDLoss` — KL divergence between channel-wise softmax distributions of teacher and student feature maps
- `MGDLoss` — spatial mask `(N,1,H,W)` with 65% positions zeroed, a generation network (`3×3 Conv → ReLU → 3×3 Conv`) per teacher channel layer, MSE loss vs raw teacher features
- `FeatureLoss` — `align_module` (`Conv2d+BN` per layer) maps student channels to teacher channel count; dispatches to `CWDLoss` or `MGDLoss`
- `DistillationLoss` — `_find_layers()` walks `model.named_modules()` to locate layers `["6","8","12","15","18","21"]` at `conv2` submodules that have a `conv` attribute; `register_hook()`/`remove_handle_()` manage forward hooks per epoch; `get_loss()` calls `distill_loss_fn` and clears output lists

---

### Modified Files

#### `yolo/tools/solver.py`

- Added `load_model_config(name)` helper — loads teacher architecture YAML from `yolo/config/model/{name}.yaml`
- `TrainModel.setup()`: when `cfg.task.teacher_weight != "None"`, calls `load_model_config`, `create_model`, then freezes teacher with `eval()` and `requires_grad=False`; passes `model_s`, `model_t`, `device` to `create_loss_function`
- Added `on_train_epoch_start()`: registers distillation hooks via `self.loss_fn.distiller.register_hook()`
- Added `on_train_epoch_end()`: removes hooks via `self.loss_fn.distiller.remove_handle_()`
- `training_step()`: added teacher forward pass inside `torch.no_grad()` to populate hooks each batch
- `configure_optimizers()`: adds `self.loss_fn.distiller.distill_loss_fn.parameters()` as a 4th optimizer param group and appends `0` to `optimizer.max_lr`

#### `yolo/tools/loss_functions.py`

- Added import: `from yolo.tools.distill_loss import DistillationLoss`
- `DualLoss.__init__()`: added `model_s`, `model_t`, `device` parameters; stores `self.total_epochs = cfg.task.epoch`; sets `self.use_distill = model_s is not None and model_t is not None`; constructs `DistillationLoss(model_s, model_t, distiller=loss_cfg.distiller_type, device=device)`
- `DualLoss.__call__()`: added `epoch_num` parameter; computes `distill_weight` (cosine warmup `0.1→1.0` for MGD, fixed `0.3` for CWD); calls `self.distiller.get_loss()`; appends `distill_weight * distill_loss` to `total_loss`; logs `Loss/DistillLoss`
- `create_loss_function()`: added `model_s=None, model_t=None, device=None` parameters passed through to `DualLoss`

#### `yolo/utils/model_utils.py`

- `create_scheduler()`: changed hardcoded `[lambda2, lambda1, lambda1]` (3 lambdas) to `[lambda2] + [lambda1] * (len(optimizer.param_groups) - 1)` — supports any number of param groups

#### `yolo/config/task/train.yaml`

- Added `teacher_weight: None` and `teacher_model: None` fields (distillation is disabled when `None`)
- Added `loss.distiller_type: cwd` default field
- Updated data augmentation defaults (batch_size, MixUp, HorizontalFlip, VerticalFlip)

#### `yolo/config/task/validation.yaml`

- `nms.min_confidence`: `0.0001` → `0.25`
- `nms.max_bbox`: `1000` → `300`

---

## References

- [Masked Generative Distillation (ECCV 2022)](https://arxiv.org/abs/2205.01529)
- [MGD Reference Implementation](https://github.com/yzd-v/MGD)
- [Channel-Wise Distillation](https://github.com/pppppM/mmdetection-distiller)
- [YOLO Distiller](https://github.com/danielsyahputra/yolo-distiller)
- [Feature Imitation](https://github.com/twangnh/Distilling-Object-Detectors)
