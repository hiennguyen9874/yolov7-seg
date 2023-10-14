import torch
import torch.nn as nn

from .onnx import *
from .trt import *


class End2End(nn.Module):
    """export onnx or tensorrt model with NMS operation."""

    def __init__(
        self,
        model,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        nc=80,
        max_wh=640,
        pooler_scale=0.25,
        device=None,
        trt=False,
    ):
        super().__init__()
        device = device if device else torch.device("cpu")
        self.model = model.to(device)
        # self.patch_model = ONNX_TRT2 if trt else ONNX_ORT
        self.patch_model = ONNX_TRT if trt else ONNX_ORT
        self.end2end = self.patch_model(
            max_obj=max_obj,
            iou_thres=iou_thres,
            score_thres=score_thres,
            nc=nc,
            max_wh=max_wh,
            pooler_scale=pooler_scale,
            device=device,
        )
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x


class End2EndRoialign(nn.Module):
    """export onnx or tensorrt model with NMS operation."""

    def __init__(
        self,
        model,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        nc=80,
        mask_resolution=56,
        max_wh=640,
        pooler_scale=0.25,
        sampling_ratio=0,
        device=None,
        trt=False,
        roi_align_type=0,
    ):
        super().__init__()
        device = device if device else torch.device("cpu")
        self.model = model.to(device)
        self.patch_model = ONNX_TRT_ROIALIGN2 if trt else ONNX_ORT_ROIALIGN
        self.end2end = self.patch_model(
            max_obj=max_obj,
            iou_thres=iou_thres,
            score_thres=score_thres,
            nc=nc,
            mask_resolution=mask_resolution,
            max_wh=max_wh,
            pooler_scale=pooler_scale,
            sampling_ratio=sampling_ratio,
            device=device,
            roi_align_type=roi_align_type,
        )
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x
