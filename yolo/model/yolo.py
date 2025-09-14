from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from omegaconf import ListConfig, OmegaConf
from torch import nn

from yolo.config.config import ModelConfig, YOLOLayer
from yolo.tools.dataset_preparation import prepare_weight
from yolo.utils.logger import logger
from yolo.utils.module_utils import get_layer_map

from yolo.model.module import Concat
from yolo.tools.anchor import make_center_anchors


class YOLO(nn.Module):
    """
    A preliminary YOLO (You Only Look Once) model class still under development.

    Parameters:
        model_cfg: Configuration for the YOLO model. Expected to define the layers,
                   parameters, and any other relevant configuration details.
    """

    def __init__(self, model_cfg: ModelConfig, class_num: int = 80):
        super(YOLO, self).__init__()
        self.num_classes = class_num
        self.layer_map = get_layer_map()  # Get the map Dict[str: Module]
        self.model: List[YOLOLayer] = nn.ModuleList()
        self.reg_max = getattr(model_cfg.anchor, "reg_max", 16)
        self.build_model(model_cfg.model)

    def build_model(self, model_arch: Dict[str, List[Dict[str, Dict[str, Dict]]]]):
        self.layer_index = {}
        output_dim, layer_idx = [3], 1
        logger.info(f":tractor: Building YOLO")
        for arch_name in model_arch:
            if model_arch[arch_name]:
                logger.info(f"  :building_construction:  Building {arch_name}")
            for layer_idx, layer_spec in enumerate(model_arch[arch_name], start=layer_idx):
                layer_type, layer_info = next(iter(layer_spec.items()))
                layer_args = layer_info.get("args", {})

                # Get input source
                source = self.get_source_idx(layer_info.get("source", -1), layer_idx)

                # Find in channels
                if any(module in layer_type for module in ["Conv", "ELAN", "ADown", "AConv", "CBLinear"]):
                    layer_args["in_channels"] = output_dim[source]
                if any(module in layer_type for module in ["Detection", "Segmentation", "Classification"]):
                    if isinstance(source, list):
                        layer_args["in_channels"] = [output_dim[idx] for idx in source]
                    else:
                        layer_args["in_channel"] = output_dim[source]
                    layer_args["num_classes"] = self.num_classes
                    layer_args["reg_max"] = self.reg_max

                # create layers
                layer = self.create_layer(layer_type, source, layer_info, **layer_args)
                self.model.append(layer)

                if layer.tags:
                    if layer.tags in self.layer_index:
                        raise ValueError(f"Duplicate tag '{layer_info['tags']}' found.")
                    self.layer_index[layer.tags] = layer_idx

                out_channels = self.get_out_channels(layer_type, layer_args, output_dim, source)
                output_dim.append(out_channels)
                setattr(layer, "out_c", out_channels)
            layer_idx += 1

    def forward(self, x, external: Optional[Dict] = None, shortcut: Optional[str] = None, target=None):
            
        if target != None:
            target_v5 = self._convert_targets_to_v5_format(target)
            preds, features = self._forward_once(x, external, shortcut, target)
            mask = self._get_imitation_mask(features, target_v5).unsqueeze(1)
            return preds, features, mask
        return self._forward_once(x, external, shortcut, target) 

    def _forward_once(self, x, external: Optional[Dict] = None, shortcut: Optional[str] = None, target=None):
        y = {0: x, **(external or {})}
        cnt = 0
        output = dict()
        for index, layer in enumerate(self.model, start=1):
            if isinstance(layer.source, list):
                model_input = [y[idx] for idx in layer.source]
            else:
                model_input = y[layer.source]

            external_input = {source_name: y[source_name] for source_name in layer.external}

            x = layer(model_input, **external_input)
            y[-1] = x

            # capture feature at 2nd Concat (if target is provided)
            if isinstance(layer, Concat):
                cnt += 1
                if cnt == 2:
                    feature = x

            if layer.usable:
                y[index] = x
            if layer.output:
                output[layer.tags] = x
                if layer.tags == shortcut:
                    # return output
                    return (output, feature) if target is not None else output
        # return output
        return (output, feature) if target is not None else output

    def _convert_targets_to_v5_format(self, targets):
        """
        YOLO-MIT targets: (B, max_obj, 5) → YOLOv5 target format: (N, 6)
        """
        device = targets.device
        B, M, _ = targets.shape
        out = []
        for b in range(B):
            # take only valid rows
            valid = targets[b][(targets[b][:, 1:] != 0).any(dim=1)]
            if valid.numel() == 0:
                continue
            # prepend batch index
            batch_idx = torch.full((valid.size(0), 1), b, device=device)
            out.append(torch.cat([batch_idx, valid], dim=1))
        return torch.cat(out, dim=0)
        # if len(out):
        #     return torch.cat(out, dim=0)
        # return targets.new_zeros((0,6))

    def _get_imitation_mask(self, x, targets, iou_factor=0.5):
        """
        gt_box: (B, K, 4) [x_min, y_min, x_max, y_max]
        """
        out_size = x.size(2)
        batch_size = x.size(0)
        device = targets.device

        mask_batch = torch.zeros([batch_size, out_size, out_size])
        
        if not len(targets):
            return mask_batch
        
        gt_boxes = [[] for i in range(batch_size)]
        for i in range(len(targets)):
            gt_boxes[int(targets[i, 0].data)] += [targets[i, 2:].clone().detach().unsqueeze(0)]
        
        max_num = 0
        for i in range(batch_size):
            max_num = max(max_num, len(gt_boxes[i]))
            if len(gt_boxes[i]) == 0:
                gt_boxes[i] = torch.zeros((1, 4), device=device)
            else:
                gt_boxes[i] = torch.cat(gt_boxes[i], 0)
        
        for i in range(batch_size):
            # print(gt_boxes[i].device)
            if max_num - gt_boxes[i].size(0):
                gt_boxes[i] = torch.cat(
                    (gt_boxes[i], torch.zeros((max_num - gt_boxes[i].size(0), 4), device=device)), 0
                )
            gt_boxes[i] = gt_boxes[i].unsqueeze(0)
                
        
        gt_boxes = torch.cat(gt_boxes, 0)
        gt_boxes *= out_size
        
        center_anchors = make_center_anchors(
            anchors_wh=self.anchors, grid_size=out_size, device=device
        )
        anchors = center_to_corner(center_anchors).view(-1, 4)  # (N, 4)
        
        gt_boxes = center_to_corner(gt_boxes)

        mask_batch = torch.zeros([batch_size, out_size, out_size], device=device)

        for i in range(batch_size):
            num_obj = gt_boxes[i].size(0)
            if not num_obj:
                continue
             
            IOU_map = find_jaccard_overlap(
                anchors, gt_boxes[i], 0).view(out_size, out_size, self.num_anchors, num_obj
            )
            max_iou, _ = IOU_map.view(-1, num_obj).max(dim=0)
            mask_img = torch.zeros([out_size, out_size], dtype=torch.int64, requires_grad=False).type_as(x)
            threshold = max_iou * iou_factor

            for k in range(num_obj):

                mask_per_gt = torch.sum(IOU_map[:, :, :, k] > threshold[k], dim=2)

                mask_img += mask_per_gt

                mask_img += mask_img
            mask_batch[i] = mask_img

        mask_batch = mask_batch.clamp(0, 1)
        return mask_batch  # (B, h, w)

    def get_out_channels(self, layer_type: str, layer_args: dict, output_dim: list, source: Union[int, list]):
        if hasattr(layer_args, "out_channels"):
            return layer_args["out_channels"]
        if layer_type == "CBFuse":
            return output_dim[source[-1]]
        if isinstance(source, int):
            return output_dim[source]
        if isinstance(source, list):
            return sum(output_dim[idx] for idx in source)

    def get_source_idx(self, source: Union[ListConfig, str, int], layer_idx: int):
        if isinstance(source, ListConfig):
            return [self.get_source_idx(index, layer_idx) for index in source]
        if isinstance(source, str):
            source = self.layer_index[source]
        if source < -1:
            source += layer_idx
        if source > 0:  # Using Previous Layer's Output
            self.model[source - 1].usable = True
        return source

    def create_layer(self, layer_type: str, source: Union[int, list], layer_info: Dict, **kwargs) -> YOLOLayer:
        if layer_type in self.layer_map:
            layer = self.layer_map[layer_type](**kwargs)
            setattr(layer, "layer_type", layer_type)
            setattr(layer, "source", source)
            setattr(layer, "in_c", kwargs.get("in_channels", None))
            setattr(layer, "output", layer_info.get("output", False))
            setattr(layer, "tags", layer_info.get("tags", None))
            setattr(layer, "external", layer_info.get("external", []))
            setattr(layer, "usable", 0)
            return layer
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def save_load_weights(self, weights: Union[Path, OrderedDict]):
        """
        Update the model's weights with the provided weights.

        args:
            weights: A OrderedDict containing the new weights.
        """
        if isinstance(weights, Path):
            weights = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)
        if "state_dict" in weights:
            weights = {name.removeprefix("model.model."): key for name, key in weights["state_dict"].items()}
        model_state_dict = self.model.state_dict()

        # TODO1: autoload old version weight
        # TODO2: weight transform if num_class difference

        error_dict = {"Mismatch": set(), "Not Found": set()}
        for model_key, model_weight in model_state_dict.items():
            if model_key not in weights:
                error_dict["Not Found"].add(tuple(model_key.split(".")[:-2]))
                continue
            if model_weight.shape != weights[model_key].shape:
                error_dict["Mismatch"].add(tuple(model_key.split(".")[:-2]))
                continue
            model_state_dict[model_key] = weights[model_key]

        for error_name, error_set in error_dict.items():
            error_dict = dict()
            for layer_idx, *layer_name in error_set:
                if layer_idx not in error_dict:
                    error_dict[layer_idx] = [".".join(layer_name)]
                else:
                    error_dict[layer_idx].append(".".join(layer_name))
            for layer_idx, layer_name in error_dict.items():
                layer_name.sort()
                logger.warning(f":warning: Weight {error_name} for Layer {layer_idx}: {', '.join(layer_name)}")

        self.model.load_state_dict(model_state_dict)


def create_model(model_cfg: ModelConfig, weight_path: Union[bool, Path] = True, class_num: int = 80) -> YOLO:
    """Constructs and returns a model from a Dictionary configuration file.

    Args:
        config_file (dict): The configuration file of the model.

    Returns:
        YOLO: An instance of the model defined by the given configuration.
    """
    OmegaConf.set_struct(model_cfg, False)
    model = YOLO(model_cfg, class_num)
    if weight_path:
        if weight_path == True:
            weight_path = Path("weights") / f"{model_cfg.name}.pt"
        elif isinstance(weight_path, str):
            weight_path = Path(weight_path)

        if not weight_path.exists():
            logger.info(f"🌐 Weight {weight_path} not found, try downloading")
            prepare_weight(weight_path=weight_path)
        if weight_path.exists():
            model.save_load_weights(weight_path)
            logger.info(":white_check_mark: Success load model & weight")
    else:
        logger.info(":white_check_mark: Success load model")
    return model
