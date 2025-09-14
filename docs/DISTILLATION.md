# YOLO Distillation Methods Documentation

This document describes the implementation of knowledge distillation (KD) methods integrated into the MIT-licensed YOLO repository.

## Implementation Details and Key changes

### Configuration (task/train.yaml)
```yaml
# Distillation settings
teacher_weight: None  # Path to teacher model weights
teacher_model: None   # Teacher model architecture

loss:
  distiller_type: cwd  # Options: cwd, mgd, feature_imitation_mask
  objective:
    BCELoss: 0.5
    BoxLoss: 7.5
    ...
```
- teacher_weight: Path to a pre-trained teacher checkpoint.
- teacher_model: Architecture of the teacher (e.g., v9-s, v9-m).
- distiller_type: Selects the distillation strategy:
    - cwd: Channel-Wise Distillation
    - mgd: Masked Generative Distillation
    - feature_imitation_mask: In progress



### Distillation Loss Implementation (CWD and MGD)

Distill Loss are implemented under [yolo/tools/distill_loss.py](yolo/tools/distill_loss.py)

Intermediate feature maps are extracted from both teacher and student models via forward hooks.
Currently, layers (6, 8, 12, 15, 18, 21) are used for feature extraction.


#### 1. Masked Generative Distillation (MGD) 

Random binary masks are applied to student feature maps.
Dynamic Weight Scheduling is used to gradually increase distillation strength, starting with weight of 0.1 and increaseing to maximum weight of 1.0.

```python
distill_weight = ((1 - math.cos(epoch * math.pi / len_train_loader)) / 2) * (1 - 0.1) + 0.1
```


#### 2. Channel-Wise Distillation (CWD)

CWD loss aligns channel-level distributions between teacher and student feature maps. KL divergence is computed per-channel across spatial dimensions.

Unlike MGD, CWD uses a fixed weight of 0.3 for stability because dynamic weight like above didn't improve the performance.

```python
distill_weight = 0.3
total_loss = [
    iou_rate * (aux_iou * aux_rate + main_iou),
    dfl_rate * (aux_dfl * aux_rate + main_dfl),
    cls_rate * (aux_cls * aux_rate + main_cls),
    distill_weight * distill_loss 
]
```


#### 3. Feature Imitation with Masking (Work in Progress)
Still under progress.

    
### Loss Implementation [yolo/tools/loss_functions.py](yolo/tools/loss_functions.py)
Distillation is wrapped in the DualLoss class, which combines YOLO’s detection loss with distillation losses.

### Distalltion training process

Distillation training process in TrainModel under [yolo/tools/solver.py](yolo/tools/solver.py)

Process:
- Teacher and student process the same input image.
- Hooks collect intermediate features.
- Distillation losses are computed (CWD, MGD, or future methods).
- Loss is added to YOLO’s detection loss.


## Usage

### Training with CWD
```shell
python yolo/lazy.py task=train \
    model=v9-t \
    dataset=coins \
    task.teacher_weight=weights/v9-s.pt \
    task.teacher_model=v9-s \
    use_wandb=True \
    task.loss.distiller_type=cwd
```

### Training with MGD
```shell
python yolo/lazy.py task=train \
    model=v9-t \
    dataset=coins \
    task.teacher_weight=weights/v9-s.pt \
    task.teacher_model=v9-s \
    use_wandb=True \
    task.loss.distiller_type=mgd
```

## References

The implementation draws inspiration from:
- [YOLOv5 Knowledge Distillation](https://github.com/wonbeomjang/yolov5-knowledge-distillation)
- [YOLO Distiller](https://github.com/danielsyahputra/yolo-distiller)
- [Channel-Wise Distillation](https://github.com/pppppM/mmdetection-distiller)
- [MGD Implementation](https://github.com/yzd-v/MGD)
- [Feature Imitation](https://github.com/twangnh/Distilling-Object-Detectors)
