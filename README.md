# YOLO Knowledge Distillation (MIT)

This repository implements knowledge distillation methods for YOLO based on the [MIT-licensed YOLO repository](https://github.com/MultimediaTechLab/YOLO). The KD methods are added with minimal modifications to the original YOLO codebase.

## Supported distillation methods:

- MGD (Masked Generative Distillation)
- CWD (Channel-Wise Distillation)
- Feature Imitation with Masking (in progress)

## Installation

```shell
git clone git@github.com:MultimediaTechLab/YOLO.git
cd YOLO
pip install -r requirements.txt
```

## Tasks

YOLO Dataset BBox format (cx cy w h) doesn't work with YOLO MIT repo. You need to convert the bbox to ( xmin ymin xmax ymax ) format. Use this script to convert your labels. 
```
python convert_labels.py
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

```shell
## Using CWD Distill Loss
python yolo/lazy.py task=train model=v9-t dataset=coins task.teacher_weight=weights/v9-s.pt task.teacher_model=v9-s use_wandb=True task.loss.distiller_type=cwd

# Using MGD Distill Loss
python yolo/lazy.py task=train model=v9-t dataset=coins task.teacher_weight=weights/v9-s.pt task.teacher_model=v9-s use_wandb=True task.loss.distiller_type=mgd

```

### Inference

To use a model for object detection, use:

```shell
python yolo/lazy.py # if cloned from GitHub
python yolo/lazy.py task=inference \ # default is inference
                    name=AnyNameYouWant \ # AnyNameYouWant
                    device=cpu \ # hardware cuda, cpu, mps
                    model=v9-s \ # model version: v9-c, m, s
                    task.nms.min_confidence=0.1 \ # nms config
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

## More Details

Check more details on Distillation implementation in [docs/DISTILLATION.md](docs/DISTILLATION.md).

## Acknowledgements

- [YOLO MIT](https://github.com/MultimediaTechLab/YOLO)
- [YOLOv5 KD Implementation](https://github.com/wonbeomjang/yolov5-knowledge-distillation)
- [Ultralytics KD Implementation](https://github.com/danielsyahputra/yolo-distiller)
- [Channel-Wise Distillation Loss Implementation](https://github.com/pppppM/mmdetection-distiller)
- [Mask Generation Distillation Loss Implementation](https://github.com/yzd-v/MGD)
- [Distilling Object Detectors with Fine-grained Feature Imitation](https://github.com/twangnh/Distilling-Object-Detectors)

