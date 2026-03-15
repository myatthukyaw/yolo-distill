# Contributing to YOLO Distillation

Thank you for your interest in contributing to this project! We value your contributions and want to make the process as easy and enjoyable as possible. Below you will find the guidelines for contributing.

Your contributions are greatly appreciated and vital to the project's success!
Please feel free to [open an issue](https://github.com/myatthukyaw/yolo-distill/issues) for questions or discussions.

## Quick Links
- [Main README](../README.md)
- [Distillation Docs](DISTILLATION.md)
- [License](../LICENSE)
- [Issue Tracker](https://github.com/myatthukyaw/yolo-distill/issues)
- [Pull Requests](https://github.com/myatthukyaw/yolo-distill/pulls)

## How to Contribute

### Proposing Enhancements
For feature requests or improvements, open an issue with:
- A clear title and description
- Why this enhancement would be useful
- Considerations or potential implementation details

### Reporting Bugs
For bug reports, open an issue with:
- A clear title and description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, PyTorch version)

## Testing and Formatting
We strive to maintain a high standard of quality in our codebase:
- **Testing:** We use `pytest` for testing. Please add tests for new code you create.
- **Formatting:** Our code follows a consistent style enforced by `isort` for imports sorting and `black` for code formatting. Run these tools to format your code before submitting a pull request.

---

## Code Changes from Original YOLO

This section documents every change made to the original YOLO repository to add distillation support.

### New Files Added

| File | Description |
|------|-------------|
| `yolo/tools/distill_loss.py` | Entire distillation loss module — see classes below |


**`yolo/tools/distill_loss.py`** contains four classes:
- `CWDLoss` — KL divergence between channel-wise softmax distributions of teacher and student feature maps
- `MGDLoss` — spatial mask `(N,1,H,W)` with 65% positions zeroed, a generation network (`3×3 Conv -> ReLU -> 3×3 Conv`) per teacher channel layer, MSE loss vs raw teacher features
- `FeatureLoss` — `align_module` (`Conv2d+BN` per layer) maps student channels to teacher channel count; dispatches to `CWDLoss` or `MGDLoss`
- `DistillationLoss` — `_find_layers()` walks `model.named_modules()` to locate layers `["6","8","12","15","18","21"]` at `conv2` submodules that have a `conv` attribute; `register_hook()`/`remove_handle_()` manage forward hooks per epoch; `get_loss()` calls `distill_loss_fn` and clears output lists

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
- `DualLoss.__call__()`: added `epoch_num` parameter; computes `distill_weight` (cosine warmup `0.1->1.0` for MGD, fixed `0.3` for CWD); calls `self.distiller.get_loss()`; appends `distill_weight * distill_loss` to `total_loss`; logs `Loss/DistillLoss`
- `create_loss_function()`: added `model_s=None, model_t=None, device=None` parameters passed through to `DualLoss`

#### `yolo/utils/model_utils.py`

- `create_scheduler()`: changed hardcoded `[lambda2, lambda1, lambda1]` (3 lambdas) to `[lambda2] + [lambda1] * (len(optimizer.param_groups) - 1)` — supports any number of param groups

#### `yolo/tools/data_augmentation.py`

- `MixUp.__call__()`: added `TF.resize(image2, [H, W])` before blending when `image2.shape != image1.shape` — fixes crash when `get_more_data()` returns images of different spatial sizes

#### `yolo/config/task/train.yaml`

- Added `teacher_weight: None` and `teacher_model: None` fields (distillation is disabled when `None`)
- Added `loss.distiller_type: cwd` default field
- Updated data augmentation defaults (batch_size, MixUp, HorizontalFlip, VerticalFlip)

#### `yolo/config/task/validation.yaml`

- `nms.min_confidence`: `0.0001` -> `0.25`
- `nms.max_bbox`: `1000` -> `300`

#### `yolo/config/config.py`

- Added `wandb_group: Optional[str] = None` to the `Config` dataclass

#### `yolo/config/general.yaml`

- Added `wandb_group: null`

#### **LR warmup lambda count mismatch (`yolo/utils/model_utils.py`)**

`create_scheduler` originally hardcoded 3 `lr_lambdas` for `LambdaLR`, matching the 3 base param groups from `create_optimizer`. With distillation, `configure_optimizers` adds a 4th group, causing:
```
ValueError: Expected 4 lr_lambdas, but got 3
```
Fix: build the lambda list dynamically from the optimizer's actual param group count:
```python
lr_lambdas = [lambda2] + [lambda1] * (len(optimizer.param_groups) - 1)
warmup_schedule = LambdaLR(optimizer, lr_lambda=lr_lambdas)
```

#### **max_lr list too short for the distill param group (`yolo/tools/solver.py`)**

`create_optimizer` sets `optimizer.max_lr = [0.1, 0, 0]` for the 3 base groups. The custom `next_batch` method iterates all param groups and indexes into `max_lr`, causing an `IndexError` when the 4th distillation group is present.
Fix: extend `max_lr` when the distillation group is added:
```python
if self.loss_fn.use_distill:
    distill_params = list(self.loss_fn.distiller.distill_loss_fn.parameters())
    optimizer.add_param_group({'params': distill_params})
    optimizer.max_lr.append(0)
```



### Distillation Network Optimizer

The generation network (`MGDLoss.generation`) and channel alignment layers (`FeatureLoss.align_module`) are **not** part of the YOLO `self.model` object. They live inside `DistillationLoss -> FeatureLoss`, a separate `nn.Module` hierarchy.

To ensure they are trained, `configure_optimizers` explicitly adds them as a 4th parameter group:

```python
def configure_optimizers(self):
    optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
    if self.loss_fn.use_distill:
        distill_params = list(self.loss_fn.distiller.distill_loss_fn.parameters())
        optimizer.add_param_group({'params': distill_params})
    scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
    return [optimizer], [scheduler]
```

In `yzd-v/MGD`, the loss modules are stored in an `nn.ModuleDict` alongside the student model, so they are automatically included when the optimizer is initialized with `model.parameters()`. This repo keeps them separate, so explicit registration is required.

The distillation networks share the same optimizer type, learning rate, and schedule as the detection backbone.
