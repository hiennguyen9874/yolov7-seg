# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Experimental modules
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.downloads import attempt_download


class Sum(nn.Module):
    # Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, n, weight=False):  # n: number of inputs
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class MixConv2d(nn.Module):
    # Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595
    def __init__(
        self, c1, c2, k=(1, 3), s=1, equal_ch=True
    ):  # ch_in, ch_out, kernel, stride, ch_strategy
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[
                0
            ].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [
                nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False)
                for k, c_ in zip(k, c_)
            ]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


class ORT_NMS(torch.autograd.Function):
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
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    ):
        return g.op(
            "NonMaxSuppression",
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        )


class ORT_ROIAlign(torch.autograd.Function):
    """ONNX-Runtime NMS operation"""

    @staticmethod
    def forward(
        ctx,
        X,
        rois,
        batch_indices,
        output_height=56,
        output_width=56,
        sampling_ratio=1,
        spatial_scale=0.25,
        # coordinate_transformation_mode="output_half_pixel",
        mode="avg",
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
        output_height,
        output_width,
        sampling_ratio,
        spatial_scale,
        # coordinate_transformation_mode="output_half_pixel",
        mode="avg",
    ):
        return g.op(
            "RoiAlign",
            X,
            rois,
            batch_indices,
            # coordinate_transformation_mode=coordinate_transformation_mode,
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
        selected_indices = ORT_NMS.apply(
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
        selected_indices = ORT_NMS.apply(
            nmsbox, max_score_tp, self.max_obj, self.iou_threshold, self.score_threshold
        )

        total_object = selected_indices.shape[0]

        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y, :]
        selected_categories = category_id[X, Y, :].float()
        selected_scores = max_score[X, Y, :]
        selected_mask = mask[X, Y, :]

        # Test1-5
        # TODO: aligned=True current not support
        pooled_proto = ORT_ROIAlign.apply(
            proto,
            selected_boxes,
            X,
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


class TRT_NMS(torch.autograd.Function):
    """TensorRT NMS operation using EfficientNMS_TRT"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        iou_threshold=0.45,
        max_output_boxes=100,
        score_activation=0,
        score_threshold=0.25,
        box_coding=1,
    ):
        device = boxes.device
        dtype = boxes.dtype
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(
            0, max_output_boxes, (batch_size, 1), device=device, dtype=torch.int32
        )
        det_boxes = torch.randn(batch_size, max_output_boxes, 4, device=device, dtype=dtype)
        det_scores = torch.randn(batch_size, max_output_boxes, device=device, dtype=dtype)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), device=device, dtype=torch.int32
        )
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        iou_threshold=0.45,
        max_output_boxes=100,
        score_activation=0,
        score_threshold=0.25,
        box_coding=1,
    ):
        out = g.op(
            "TRT::EfficientNMS_TRT",
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=4,
        )
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class TRT_NMS2(torch.autograd.Function):
    """TensorRT NMS operation using EfficientNMS_ONNX_TRT"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        score_threshold=0.25,
        iou_threshold=0.45,
        max_output_boxes_per_class=100,
    ):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,), device=device).sort()[0]
        idxs = torch.arange(100, 100 + num_det, device=device)
        zeros = torch.zeros((num_det,), dtype=torch.int64, device=device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return_selected_indices = torch.zeros(
            (batch * max_output_boxes_per_class, 3), dtype=torch.int64, device=device
        )
        return_selected_indices[: selected_indices.shape[0]] = selected_indices
        return return_selected_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        score_threshold=0.25,
        iou_threshold=0.45,
        max_output_boxes_per_class=100,
    ):
        return g.op(
            "TRT::EfficientNMS_ONNX_TRT",
            boxes,
            scores,
            score_threshold_f=score_threshold,
            iou_threshold_f=iou_threshold,
            max_output_boxes_per_class_i=max_output_boxes_per_class,
            center_point_box_i=0,
            outputs=1,
        )


class TRT_NMS3(torch.autograd.Function):
    """ONNX-Runtime NMS operation using NonMaxSuppression"""

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
        return_selected_indices = torch.zeros(
            (batch * max_output_boxes_per_class, 3), dtype=torch.int64, device=device
        )
        return_selected_indices[: selected_indices.shape[0]] = selected_indices
        return return_selected_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
    ):
        return g.op(
            "NonMaxSuppression",
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        )


class TRT_NMS4(torch.autograd.Function):
    """TensorRT NMS operation using EfficientNMSCustom_TRT"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        iou_threshold=0.45,
        max_output_boxes=100,
        score_activation=0,
        score_threshold=0.25,
        box_coding=1,
    ):
        device = boxes.device
        dtype = boxes.dtype

        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(
            0, max_output_boxes, (batch_size, 1), device=device, dtype=torch.int32
        )
        det_boxes = torch.randn(batch_size, max_output_boxes, 4, device=device, dtype=dtype)
        det_scores = torch.randn(batch_size, max_output_boxes, device=device, dtype=dtype)
        det_classes = torch.randint(
            0, num_classes, (batch_size, max_output_boxes), device=device, dtype=torch.int32
        )
        det_indices = torch.randint(
            0,
            num_boxes,
            (batch_size, max_output_boxes),
            device=device,
            dtype=torch.int32,
        )
        return num_det, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        iou_threshold=0.45,
        max_output_boxes=100,
        score_activation=0,
        score_threshold=0.25,
        box_coding=1,
    ):
        out = g.op(
            "TRT::EfficientNMSCustom_TRT",
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=5,
        )
        num_det, det_boxes, det_scores, det_classes, det_indices = out
        return num_det, det_boxes, det_scores, det_classes, det_indices


class TRT_ROIAlign(torch.autograd.Function):
    """TensorRT RoiAlign operation"""

    @staticmethod
    def forward(
        ctx,
        X,
        rois,
        output_height,
        output_width,
        spatial_scale,
        sampling_ratio,
        aligned=1,
        mode="avg",
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
        output_height,
        output_width,
        spatial_scale,
        sampling_ratio,
        aligned=1,
        mode="avg",
    ):
        return g.op(
            "TRT::RoIAlignDynamic_TRT",
            X,
            rois,
            output_height_i=output_height,
            output_width_i=output_width,
            spatial_scale_f=spatial_scale,
            sampling_ratio_i=sampling_ratio,
            mode_s=mode,
            aligned_i=aligned,
            outputs=1,
        )


class TRT_ROIAlign2(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        feature_map,
        roi,
        pooled_size=56,
        image_size=640,
        sampling_ratio=1,
        roi_coords_absolute=1,
        roi_coords_swap=0,
        roi_coords_transform=2,
        legacy=0,
    ):
        device = roi.device
        dtype = roi.dtype
        ROI_N, ROI_R, ROI_D = roi.shape
        F_N, F_C, F_H, F_W = feature_map.shape
        assert ROI_N == F_N
        return torch.randn(
            (ROI_N, ROI_R, F_C, pooled_size, pooled_size), device=device, dtype=dtype
        )

    @staticmethod
    def symbolic(
        g,
        feature_map,
        roi,
        pooled_size=56,
        image_size=640,
        sampling_ratio=1,
        roi_coords_absolute=1,
        roi_coords_swap=0,
        roi_coords_transform=2,
        legacy=0,
    ):
        return g.op(
            "TRT::RoIAlign2Dynamic_TRT",
            feature_map,
            roi,
            pooled_size_i=pooled_size,
            sampling_ratio_i=sampling_ratio,
            roi_coords_absolute_i=roi_coords_absolute,
            roi_coords_swap_i=roi_coords_swap,
            roi_coords_transform_i=roi_coords_transform,
            image_size_i=image_size,
            legacy_i=legacy,
            outputs=1,
        )


class ONNX_TRT(nn.Module):
    """onnx module with ONNX-TensorRT NMS operation."""

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
        self.max_obj_i = max_obj
        self.max_wh = max_wh
        self.device = device if device else torch.device("cpu")
        self.nc = nc
        self.pooler_scale = pooler_scale

        # For TRT_NMS
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

        selected_indices = TRT_NMS3.apply(
            nmsbox,
            max_score_tp,
            self.max_obj,
            self.iou_threshold,
            self.score_threshold,
        ).to(torch.long)

        total_object = selected_indices.shape[0]

        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y, :]
        selected_categories = category_id[X, Y, :].float()
        selected_scores = max_score[X, Y, :]
        selected_mask = mask[X, Y, :]

        # # Test0
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

        # Upsample
        masks = masks.view(total_object, proto_h * proto_w)

        # X = X.unsqueeze(1).float()

        # If sum(axis=1) is zero
        num_object1 = (
            torch.topk(
                torch.where(
                    selected_indices.sum(dim=1) > 0,
                    torch.arange(0, total_object, 1, device=self.device, dtype=torch.int32),
                    torch.zeros(total_object, device=self.device, dtype=torch.int32),
                ).to(torch.float),
                k=1,
                largest=True,
            )[1]
            + 1
        ).reshape((1,))

        # If lag not change
        selected_indices_lag = (selected_indices[1:] - selected_indices[:-1]).sum(dim=1)
        num_object2 = (
            torch.topk(
                torch.where(
                    selected_indices_lag != 0,
                    torch.arange(0, total_object - 1, device=self.device, dtype=torch.int32),
                    torch.zeros((1,), device=self.device, dtype=torch.int32),
                ).to(torch.float),
                k=1,
                largest=True,
            )[1]
            + 2
        ).reshape((1,))

        num_object = (selected_indices_lag.sum() != 0).to(torch.float32) * torch.min(
            num_object1, num_object2
        )

        batch_indices_per_batch = torch.where(
            (
                X.unsqueeze(dim=1)
                == torch.arange(0, batch_size, dtype=X.dtype, device=self.device).unsqueeze(dim=0)
            )
            & torch.where(
                torch.arange(0, total_object, device=self.device, dtype=torch.int32) < num_object,
                torch.ones((1,), device=self.device, dtype=torch.int32),
                torch.zeros((1,), device=self.device, dtype=torch.int32),
            )
            .to(torch.bool)
            .unsqueeze(dim=1),
            torch.ones((1,), device=self.device, dtype=torch.int32),
            torch.zeros((1,), device=self.device, dtype=torch.int32),
        )

        num_det = batch_indices_per_batch.sum(dim=0).view(batch_size, 1).to(torch.int32)

        idxs = (
            torch.topk(
                batch_indices_per_batch.to(torch.float32)
                * torch.arange(0, total_object, dtype=torch.int32, device=self.device).unsqueeze(
                    dim=1
                ),
                k=self.max_obj_i,
                dim=0,
                largest=True,
                sorted=True,
            )[0]
            .t()
            .contiguous()
            .view(-1)
            .to(torch.long)
        )

        det_boxes = selected_boxes[idxs].view(batch_size, self.max_obj_i, 4).to(torch.float32)
        det_scores = selected_scores[idxs].view(batch_size, self.max_obj_i, 1).to(torch.float32)
        det_classes = (
            selected_categories[idxs].view(batch_size, self.max_obj_i, 1).to(torch.float32)
        )
        det_masks = (
            masks[idxs].view(batch_size, self.max_obj_i, proto_h * proto_w).to(torch.float32)
        )
        return num_det, det_boxes, det_scores, det_classes, det_masks


class ONNX_TRT2(nn.Module):
    """onnx module with ONNX-TensorRT NMS operation."""

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
        self.max_wh = max_wh
        self.device = device if device else torch.device("cpu")
        self.nc = nc
        self.pooler_scale = pooler_scale

        self.max_obj = max_obj
        self.iou_threshold = iou_thres
        self.score_threshold = score_thres

        self.background_class = (-1,)
        self.score_activation = 0

    def forward(self, x):
        boxes = x[0][:, :, :4]
        conf = x[0][:, :, 4:5]
        scores = x[0][:, :, 5 : 5 + self.nc]
        proto = x[1]
        batch_size, nm, proto_h, proto_w = proto.shape
        total_object = batch_size * self.max_obj
        masks = x[0][:, :, 5 + self.nc : 5 + self.nc + nm]

        scores *= conf

        num_det, det_boxes, det_scores, det_classes, det_indices = TRT_NMS4.apply(
            boxes,
            scores,
            self.background_class,
            self.iou_threshold,
            self.max_obj,
            self.score_activation,
            self.score_threshold,
        )
        batch_indices = torch.ones_like(det_indices) * torch.arange(
            batch_size, device=self.device, dtype=torch.int32
        ).unsqueeze(1)
        batch_indices = batch_indices.view(total_object).to(torch.long)
        det_indices = det_indices.view(total_object).to(torch.long)
        det_masks = masks[batch_indices, det_indices]
        masks = (
            (
                torch.matmul(
                    det_masks.unsqueeze(dim=1),
                    proto[batch_indices].view(total_object, nm, proto_h * proto_w),
                )
            )
            .sigmoid()
            .view(total_object, proto_h, proto_w)
        )
        downsampled_bboxes = det_boxes.view(total_object, 4) * self.pooler_scale
        x1, y1, x2, y2 = torch.chunk(downsampled_bboxes.unsqueeze(dim=2), 4, 1)
        r = torch.arange(proto_w, device=masks.device, dtype=torch.float32)[None, None, :]
        c = torch.arange(proto_h, device=masks.device, dtype=torch.float32)[None, :, None]
        masks = masks * (
            (r >= x1).to(torch.float32)
            * (r < x2).to(torch.float32)
            * (c >= y1).to(torch.float32)
            * (c < y2).to(torch.float32)
        )
        masks = masks.view(batch_size, self.max_obj, proto_h * proto_w)
        return num_det, det_boxes, det_scores, det_classes, masks


class ONNX_TRT_ROIALIGN(nn.Module):
    """onnx module with ONNX-TensorRT NMS operation."""

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
        self.max_obj_i = max_obj
        self.max_wh = max_wh
        self.device = device if device else torch.device("cpu")

        # For TRT_NMS
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

        self.nc = nc
        self.mask_resolution = mask_resolution
        self.pooler_scale = pooler_scale
        self.sampling_ratio = sampling_ratio

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

        selected_indices = TRT_NMS.apply(
            nmsbox,
            max_score_tp,
            self.max_obj,
            self.iou_threshold,
            self.score_threshold,
        ).to(torch.long)

        total_object = selected_indices.shape[0]

        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y, :]
        selected_categories = category_id[X, Y, :].float()
        selected_scores = max_score[X, Y, :]
        selected_mask = mask[X, Y, :]

        # Test1-5
        pooled_proto = TRT_ROIAlign.apply(
            proto,
            torch.cat((X.unsqueeze(1).float(), selected_boxes), dim=1),
            self.mask_resolution,
            self.mask_resolution,
            self.pooler_scale,
            self.sampling_ratio,
        )

        masks = (
            torch.matmul(
                selected_mask.unsqueeze(dim=1),
                pooled_proto.view(total_object, nm, self.mask_resolution * self.mask_resolution),
            )
            .sigmoid()
            .view(-1, self.mask_resolution * self.mask_resolution)
        )

        # X = X.unsqueeze(1).float()

        # If sum(axis=1) is zero
        num_object1 = (
            torch.topk(
                torch.where(
                    selected_indices.sum(dim=1) > 0,
                    torch.arange(0, total_object, 1, device=self.device, dtype=torch.int32),
                    torch.zeros(total_object, device=self.device, dtype=torch.int32),
                ).to(torch.float),
                k=1,
                largest=True,
            )[1]
            + 1
        ).reshape((1,))

        # If lag not change
        selected_indices_lag = (selected_indices[1:] - selected_indices[:-1]).sum(dim=1)
        num_object2 = (
            torch.topk(
                torch.where(
                    selected_indices_lag != 0,
                    torch.arange(0, total_object - 1, device=self.device, dtype=torch.int32),
                    torch.zeros((1,), device=self.device, dtype=torch.int32),
                ).to(torch.float),
                k=1,
                largest=True,
            )[1]
            + 2
        ).reshape((1,))

        num_object = (selected_indices_lag.sum() != 0).to(torch.float32) * torch.min(
            num_object1, num_object2
        )

        batch_indices_per_batch = torch.where(
            (
                X.unsqueeze(dim=1)
                == torch.arange(0, batch_size, dtype=X.dtype, device=self.device).unsqueeze(dim=0)
            )
            & torch.where(
                torch.arange(0, total_object, device=self.device, dtype=torch.int32) < num_object,
                torch.ones((1,), device=self.device, dtype=torch.int32),
                torch.zeros((1,), device=self.device, dtype=torch.int32),
            )
            .to(torch.bool)
            .unsqueeze(dim=1),
            torch.ones((1,), device=self.device, dtype=torch.int32),
            torch.zeros((1,), device=self.device, dtype=torch.int32),
        )

        num_det = batch_indices_per_batch.sum(dim=0).view(batch_size, 1).to(torch.int32)

        idxs = (
            torch.topk(
                batch_indices_per_batch.to(torch.float32)
                * torch.arange(0, total_object, dtype=torch.int32, device=self.device).unsqueeze(
                    dim=1
                ),
                k=self.max_obj_i,
                dim=0,
                largest=True,
                sorted=True,
            )[0]
            .t()
            .contiguous()
            .view(-1)
            .to(torch.long)
        )

        det_boxes = selected_boxes[idxs].view(batch_size, self.max_obj_i, 4).to(torch.float32)
        det_scores = selected_scores[idxs].view(batch_size, self.max_obj_i, 1).to(torch.float32)
        det_classes = (
            selected_categories[idxs].view(batch_size, self.max_obj_i, 1).to(torch.float32)
        )
        det_masks = (
            masks[idxs]
            .view(batch_size, self.max_obj_i, self.mask_resolution * self.mask_resolution)
            .to(torch.float32)
        )

        return num_det, det_boxes, det_scores, det_classes, det_masks


class ONNX_TRT_ROIALIGN2(nn.Module):
    """onnx module with ONNX-TensorRT NMS operation."""

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
        self.max_obj = max_obj
        self.iou_threshold = iou_thres
        self.score_threshold = score_thres
        self.nc = nc
        self.mask_resolution = mask_resolution
        self.pooler_scale = pooler_scale
        self.sampling_ratio = sampling_ratio
        self.image_size = max_wh
        self.roi_align_type = 2  # 1, or 2

        self.background_class = (-1,)
        self.score_activation = 0

    def forward(self, x):
        boxes = x[0][:, :, :4]
        conf = x[0][:, :, 4:5]
        scores = x[0][:, :, 5 : 5 + self.nc]
        proto = x[1]
        batch_size, nm, proto_h, proto_w = proto.shape
        total_object = batch_size * self.max_obj
        masks = x[0][:, :, 5 + self.nc : 5 + self.nc + nm]
        scores *= conf

        num_det, det_boxes, det_scores, det_classes, det_indices = TRT_NMS4.apply(
            boxes,
            scores,
            self.background_class,
            self.iou_threshold,
            self.max_obj,
            self.score_activation,
            self.score_threshold,
        )
        batch_indices = torch.ones_like(det_indices) * torch.arange(
            batch_size, device=self.device, dtype=torch.int32
        ).unsqueeze(1)
        batch_indices = batch_indices.view(total_object).to(torch.long)
        det_indices = det_indices.view(total_object).to(torch.long)
        det_masks = masks[batch_indices, det_indices]

        if self.roi_align_type == 1:
            pooled_proto = TRT_ROIAlign.apply(
                proto,
                torch.cat(
                    (batch_indices.unsqueeze(1).float(), det_boxes.view(total_obejct, 4)), dim=2
                ),
                self.mask_resolution,
                self.mask_resolution,
                self.pooler_scale,
                self.sampling_ratio,
            )
        else:
            pooled_proto = TRT_ROIAlign2.apply(
                proto,
                det_boxes,
                self.mask_resolution,
                self.image_size,
            )

        pooled_proto = pooled_proto.view(
            total_object, nm, self.mask_resolution * self.mask_resolution
        )

        masks = (
            torch.matmul(det_masks.unsqueeze(dim=1), pooled_proto)
            .sigmoid()
            .view(batch_size, self.max_obj, self.mask_resolution * self.mask_resolution)
        )

        return num_det, det_boxes, det_scores, det_classes, masks


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
        self.patch_model = ONNX_TRT2 if trt else ONNX_ORT
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
        )
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x


def attempt_load(weights, device=None, inplace=True, fuse=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    from models.yolo import Detect, Model, IDetect

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(
            ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval()
        )  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, IDetect, Model):
            m.inplace = inplace  # torch 1.7.0 compatibility
            if (t is Detect or t is IDetect) and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[
        torch.argmax(torch.tensor([m.stride.max() for m in model])).int()
    ].stride  # max stride
    assert all(
        model[0].nc == m.nc for m in model
    ), f"Models have different class counts: {[m.nc for m in model]}"
    return model
