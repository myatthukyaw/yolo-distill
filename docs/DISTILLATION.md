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

> For a full list of files modified from the original YOLO repo, see [CONTRIBUTING.md](CONTRIBUTING.md#code-changes-from-original-yolo).

## References

- [Masked Generative Distillation (ECCV 2022)](https://arxiv.org/abs/2205.01529)
- [MGD Reference Implementation](https://github.com/yzd-v/MGD)
- [Channel-Wise Distillation](https://github.com/pppppM/mmdetection-distiller)
- [YOLO Distiller](https://github.com/danielsyahputra/yolo-distiller)
- [Feature Imitation](https://github.com/twangnh/Distilling-Object-Detectors)
