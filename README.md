# YOLO Knowledge Distillation (MIT)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

This repository implements knowledge distillation methods for YOLO based on the [MIT-licensed YOLO repository](https://github.com/MultimediaTechLab/YOLO). The KD methods are added with minimal modifications to the original YOLO codebase.

## Supported distillation methods:

- MGD (Masked Generative Distillation)
- CWD (Channel-Wise Distillation)
- Feature Imitation with Masking (in progress)

## Results

Tested on [Face Detection Dataset](https://universe.roboflow.com/mohamed-traore-2ekkp/face-detection-mik1i/dataset/25) (Roboflow, ~4k images, 1 class).

Image size 640×640, trained for 50 epochs. 

| Model           | Params  | mAP@0.50      | mAP@0.50:0.95 |
|-----------------|--------:|:-------------:|:-------------:|
| v9-s (teacher)  | 9.7M    | **79.73**     | **46.13**     | 
| v9-t (baseline) | 2.7M    | 70.57         | 39.57         | 
| v9-t + CWD      | 2.7M    | 76.37         | 42.98         | 
| v9-t + MGD      | 2.7M    | **79.41**     | **44.53**     | 


- Baseline gap vs teacher: **-9.16 mAP@0.50**.
- CWD recovers **+5.80** over baseline closing ~63% of the gap.
- MGD recovers **+8.84** over baseline, almost fully closes the gap.
- MGD at 2.7M params matches a 9.7M teacher — **3.5× compression with <0.5% mAP loss**
- The same for mAP@0.5:0.95.


## Installation

```shell
git clone git@github.com:myatthukyaw/yolo-distill.git
cd YOLO
pip install -r requirements.txt
```

## Tasks

### Dataset Preparation

This repo (YOLO MIT) expects labels in **corner format** (`class xmin ymin xmax ymax`, normalized to [0,1]).

**Roboflow datasets** are in YOLO format - `cx cy w h`.

Use `prepare_roboflow_dataset.py`:

```shell
# Converts labels in-place (xywh to xyxy) and writes train.txt / val.txt index files
python scripts/prepare_roboflow_dataset.py --root /path/to/roboflow-dataset
```

If you have **other YOLO-format datasets** with `images/<phase>/ + labels/<phase>/` layout, use `convert_labels.py`:

```shell
python scripts/convert_labels.py --root /path/to/dataset --phase train
```

**Verify your label format** visually before training with `visualize_labels.py`:

```shell
# Side-by-side comparison of xyxy vs xywh - helps identify which format your labels are in
python scripts/visualize_labels.py --folder path/to/images/
```

### Training from scratch

To train YOLO on your machine/dataset:

1. Modify the configuration file `yolo/config/dataset/**.yaml` to point to your dataset.
2. Run the training script:

```shell
python yolo/lazy.py task=train dataset=** use_wandb=True
python yolo/lazy.py task=train task.data.batch_size=8 model=v9-c weight=False # or more args
```

### Transfer Learning

To perform transfer learning with YOLOv9:

```shell
python yolo/lazy.py task=train task.data.batch_size=8 model=v9-c dataset={dataset_config} device={cpu, mps, cuda}
```

### Distillation

CWD and MGD are **feature-level** distillation methods — they distill intermediate backbone/neck feature maps, not detection outputs. For the teacher to provide task-specific knowledge, it should be trained on the same dataset as the student first.

**Step 1 — Train the teacher on your dataset:**

```shell
python yolo/lazy.py task=train model=v9-s dataset=coins name=v9s-teacher use_wandb=True
```

**Step 2 — Train the student with distillation using the trained teacher:**

```shell
# CWD distillation
python yolo/lazy.py task=train model=v9-t dataset=coins name=v9t-cwd \
    task.teacher_weight=runs/train/v9s-teacher/.../checkpoints/epoch=49.ckpt \
    task.teacher_model=v9-s \
    task.loss.distiller_type=cwd \
    use_wandb=True

# MGD distillation
python yolo/lazy.py task=train model=v9-t dataset=coins name=v9t-mgd \
    task.teacher_weight=runs/train/v9s-teacher/.../checkpoints/epoch=49.ckpt \
    task.teacher_model=v9-s \
    task.loss.distiller_type=mgd \
    use_wandb=True
```

**Or run the full comparison automatically (trains teacher first):**

```shell
python scripts/run_comparison.py --dataset [dataset yaml filename] --epochs 50 --teacher-epochs 50

python scripts/run_comparison.py --dataset face --epochs 50 --teacher-epochs 50
```

### Inference

To run inference, use:

```shell
python yolo/lazy.py # if cloned from GitHub
python yolo/lazy.py task=inference \ # default is inference
                    name=AnyNameYouWant \ # AnyNameYouWant
                    device=cpu \ # hardware cuda, cpu, mps
                    model=v9-s \ # model version: v9-c, m, s
                    task.nms.min_confidence=0.5 \ # nms config
                    task.fast_inference=onnx \ # onnx, trt, deploy
                    task.data.source=data/toy/images/train \ # file, dir, webcam
                    +quite=True \ # Quite Output
```

### Validation

To validate model performance, or generate a json file in COCO format:

```shell
python yolo/lazy.py task=validation
python yolo/lazy.py task=validation dataset=toy
```

---

## More Details

Check more details on Distillation implementation in [docs/DISTILLATION.md](docs/DISTILLATION.md).

## Acknowledgements

- [YOLO MIT](https://github.com/MultimediaTechLab/YOLO)
- [YOLOv5 KD Implementation](https://github.com/wonbeomjang/yolov5-knowledge-distillation)
- [Ultralytics KD Implementation](https://github.com/danielsyahputra/yolo-distiller)
- [Channel-Wise Distillation Loss Implementation](https://github.com/pppppM/mmdetection-distiller)
- [Mask Generation Distillation Loss Implementation](https://github.com/yzd-v/MGD)
- [Distilling Object Detectors with Fine-grained Feature Imitation](https://github.com/twangnh/Distilling-Object-Detectors)

