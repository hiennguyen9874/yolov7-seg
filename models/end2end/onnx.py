import math
import random

import numpy as np
import torch
import torch.nn as nn


class ORT_NonMaxSuppression(torch.autograd.Function):
    """ONNX-Runtime NMS operation"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        max_output_boxes_per_class=torch.tensor([100]),
        iou_threshold=torch.tensor([0.45]),
        score_threshold=torch.tensor([0.25]),
    ):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,), device=device).sort()[0]
        idxs = torch.arange(100, 100 + num_det, device=device)
        zeros = torch.zeros((num_det,), dtype=torch.int64, device=device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        max_output_boxes_per_class=torch.tensor([100]),
        iou_threshold=torch.tensor([0.45]),
        score_threshold=torch.tensor([0.25]),
    ):
        return g.op(
            "NonMaxSuppression",
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            center_point_box_i=0,
        )


class ORT_RoiAlign(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode="half_pixel",
        mode="avg",
        output_height=56,
        output_width=56,
        sampling_ratio=1,
        spatial_scale=0.25,
    ):
        device = rois.device
        dtype = rois.dtype
        N, C, H, W = X.shape
        num_rois = rois.shape[0]
        return torch.randn((num_rois, C, output_height, output_width), device=device, dtype=dtype)

    @staticmethod
    def symbolic(
        g,
        X,
        rois,
        batch_indices,
        coordinate_transformation_mode="half_pixel",
        mode="avg",
        output_height=56,
        output_width=56,
        sampling_ratio=0,
        spatial_scale=0.25,
    ):
        return g.op(
            "RoiAlign",
            X,
            rois,
            batch_indices,
            coordinate_transformation_mode_s=coordinate_transformation_mode,
            mode_s=mode,
            output_height_i=output_height,
            output_width_i=output_width,
            sampling_ratio_i=sampling_ratio,
            spatial_scale_f=spatial_scale,
        )


class ONNX_ORT(nn.Module):
    """onnx module with ONNX-Runtime NMS operation."""

    def __init__(
        self,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        nc=80,
        max_wh=640,
        pooler_scale=0.25,
        device=None,
    ):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_wh = max_wh
        self.nc = nc
        self.pooler_scale = pooler_scale

        self.register_buffer("max_obj", torch.tensor([max_obj]))
        self.register_buffer("iou_threshold", torch.tensor([iou_thres]))
        self.register_buffer("score_threshold", torch.tensor([score_thres]))
        self.register_buffer(
            "convert_matrix",
            torch.tensor(
                [
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [-0.5, 0, 0.5, 0],
                    [0, -0.5, 0, 0.5],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        boxes = x[0][:, :, :4]
        conf = x[0][:, :, 4:5]
        scores = x[0][:, :, 5 : 5 + self.nc]
        proto = x[1]
        batch_size, nm, proto_h, proto_w = proto.shape
        mask = x[0][:, :, 5 + self.nc : 5 + self.nc + nm]
        scores *= conf
        boxes @= self.convert_matrix
        max_score, category_id = scores.max(2, keepdim=True)
        dis = category_id.float() * self.max_wh
        nmsbox = boxes + dis
        max_score_tp = max_score.transpose(1, 2).contiguous()
        selected_indices = ORT_NonMaxSuppression.apply(
            nmsbox,
            max_score_tp,
            self.max_obj,
            self.iou_threshold,
            self.score_threshold,
        )

        total_object = selected_indices.shape[0]

        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y, :]
        selected_categories = category_id[X, Y, :].float()
        selected_scores = max_score[X, Y, :]
        selected_mask = mask[X, Y, :]

        # Test0
        masks = (
            (
                torch.matmul(
                    selected_mask.unsqueeze(dim=1),
                    proto[X.to(torch.long)].view(total_object, nm, proto_h * proto_w),
                )
            )
            .sigmoid()
            .view(total_object, proto_h, proto_w)
        )

        # Crop
        downsampled_bboxes = selected_boxes * self.pooler_scale
        x1, y1, x2, y2 = torch.chunk(downsampled_bboxes.unsqueeze(dim=2), 4, 1)
        r = torch.arange(proto_w, device=masks.device, dtype=torch.float32)[None, None, :]
        # rows shape(1,w,1)
        c = torch.arange(proto_h, device=masks.device, dtype=torch.float32)[None, :, None]
        # cols shape(h,1,1)
        masks = masks * (
            (r >= x1).to(torch.float32)
            * (r < x2).to(torch.float32)
            * (c >= y1).to(torch.float32)
            * (c < y2).to(torch.float32)
        )

        X = X.unsqueeze(1).float()
        masks = masks.view(total_object, proto_h * proto_w)
        return torch.cat([X, selected_boxes, selected_categories, selected_scores, masks], 1)


class ONNX_ORT_ROIALIGN(nn.Module):
    """onnx module with ONNX-Runtime NMS operation."""

    def __init__(
        self,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        nc=80,
        mask_resolution=56,
        max_wh=640,
        pooler_scale=0.25,
        sampling_ratio=0,
        device=None,
    ):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_wh = max_wh
        self.nc = nc
        self.mask_resolution = mask_resolution
        self.pooler_scale = pooler_scale
        self.sampling_ratio = sampling_ratio

        self.register_buffer("max_obj", torch.tensor([max_obj]))
        self.register_buffer("iou_threshold", torch.tensor([iou_thres]))
        self.register_buffer("score_threshold", torch.tensor([score_thres]))
        self.register_buffer(
            "convert_matrix",
            torch.tensor(
                [
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [-0.5, 0, 0.5, 0],
                    [0, -0.5, 0, 0.5],
                ],
                dtype=torch.float32,
            ),
        )
        # self.register_buffer(
        #     "crop_size",
        #     torch.tensor([mask_resolution, mask_resolution], dtype=torch.int32),
        # )

    def forward(self, x):
        boxes = x[0][:, :, :4]
        conf = x[0][:, :, 4:5]
        scores = x[0][:, :, 5 : 5 + self.nc]
        proto = x[1]
        batch_size, nm, proto_h, proto_w = proto.shape
        mask = x[0][:, :, 5 + self.nc : 5 + self.nc + nm]
        scores *= conf
        boxes @= self.convert_matrix
        max_score, category_id = scores.max(2, keepdim=True)
        dis = category_id.float() * self.max_wh
        nmsbox = boxes + dis
        max_score_tp = max_score.transpose(1, 2).contiguous()
        selected_indices = ORT_NonMaxSuppression.apply(
            nmsbox,
            max_score_tp,
            self.max_obj,
            self.iou_threshold,
            self.score_threshold,
        )

        total_object = selected_indices.shape[0]

        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y, :]
        selected_categories = category_id[X, Y, :].float()
        selected_scores = max_score[X, Y, :]
        selected_mask = mask[X, Y, :]

        # Test1-5
        # TODO: aligned=True current not support
        pooled_proto = ORT_RoiAlign.apply(
            proto,
            selected_boxes,
            X,
            "half_pixel",
            "avg",
            self.mask_resolution,
            self.mask_resolution,
            self.sampling_ratio,
            self.pooler_scale,
        )

        masks = (
            torch.matmul(
                selected_mask.unsqueeze(dim=1),
                pooled_proto.view(total_object, nm, self.mask_resolution * self.mask_resolution),
            )
            .sigmoid()
            .view(-1, self.mask_resolution * self.mask_resolution)
        )

        X = X.unsqueeze(1).float()

        return torch.cat([X, selected_boxes, selected_categories, selected_scores, masks], 1)
