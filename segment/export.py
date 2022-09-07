# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Export a YOLOv5 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit

Format                      | `export.py --include`         | Model
---                         | ---                           | ---
PyTorch                     | -                             | yolov5s.pt
TorchScript                 | `torchscript`                 | yolov5s.torchscript
ONNX                        | `onnx`                        | yolov5s.onnx
OpenVINO                    | `openvino`                    | yolov5s_openvino_model/
TensorRT                    | `engine`                      | yolov5s.engine
CoreML                      | `coreml`                      | yolov5s.mlmodel
TensorFlow SavedModel       | `saved_model`                 | yolov5s_saved_model/
TensorFlow GraphDef         | `pb`                          | yolov5s.pb
TensorFlow Lite             | `tflite`                      | yolov5s.tflite
TensorFlow Edge TPU         | `edgetpu`                     | yolov5s_edgetpu.tflite
TensorFlow.js               | `tfjs`                        | yolov5s_web_model/

Requirements:
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime openvino-dev tensorflow-cpu  # CPU
    $ pip install -r requirements.txt coremltools onnx onnx-simplifier onnxruntime-gpu openvino-dev tensorflow  # GPU

Usage:
    $ python export.py --weights yolov5s.pt --include torchscript onnx openvino engine coreml tflite ...

Inference:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s.xml                # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov5s_web_model public/yolov5s_web_model
    $ npm start
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load, End2End, End2EndRoialign
from models.yolo import Detect, IDetect
from utils.dataloaders import LoadImages
from utils.general import (
    LOGGER,
    Profile,
    check_dataset,
    check_img_size,
    check_requirements,
    check_version,
    check_yaml,
    colorstr,
    file_size,
    get_default_args,
    print_args,
    url2file,
)
from utils.torch_utils import select_device, smart_inference_mode


def export_formats():
    # YOLOv5 export formats
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlmodel", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def try_export(inner_func):
    # YOLOv5 export decorator, i..e @try_export
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        prefix = inner_args["prefix"]
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(
                f"{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)"
            )
            return f, model
        except Exception as e:
            LOGGER.info(f"{prefix} export failure ❌ {dt.t:.1f}s: {e}")
            return None, None

    return outer_func


@try_export
def export_onnx(
    model,
    im,
    file,
    opset,
    train,
    dynamic,
    simplify,
    dynamic_batch,
    end2end,
    trt,
    topk_all,
    device,
    iou_thres,
    score_thres,
    mask_resolution,
    pooler_scale,
    sampling_ratio,
    image_size,
    cleanup,
    roi_align,
    prefix=colorstr("ONNX:"),
):
    # YOLOv5 ONNX export
    check_requirements(("onnx",))
    import onnx

    LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = file.with_suffix(".onnx")

    output_names = ["output", "proto"]
    dynamic_axes = None

    if dynamic:
        raise NotImplementedError

    if end2end:
        if trt:
            output_names = [
                "num_dets",
                "det_boxes",
                "det_scores",
                "det_classes",
                "det_masks",
            ]
        else:
            output_names = ["output"]

    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"},
        }
        output_axes = {
            "output": {0: "batch"},
            "proto": {0: "batch"},
        }

        if end2end:
            if trt:
                output_axes = {
                    "num_dets": {0: "batch"},
                    "det_boxes": {0: "batch"},
                    "det_scores": {0: "batch"},
                    "det_classes": {0: "batch"},
                    "det_masks": {0: "batch"},
                }
            else:
                output_axes = {
                    "output": {0: "num_dets"},
                }
        dynamic_axes.update(output_axes)

    if end2end:
        if roi_align:
            model = End2EndRoialign(
                model=model,
                max_obj=topk_all,
                iou_thres=iou_thres,
                score_thres=score_thres,
                nc=len(model.names),
                mask_resolution=mask_resolution,
                pooler_scale=pooler_scale,
                sampling_ratio=sampling_ratio,
                device=device,
                trt=trt,
                max_wh=max(image_size),
            )
        else:
            model = End2End(
                model=model,
                max_obj=topk_all,
                iou_thres=iou_thres,
                score_thres=score_thres,
                nc=len(model.names),
                pooler_scale=pooler_scale,
                device=device,
                trt=trt,
                max_wh=max(image_size),
            )

    torch.onnx.export(
        model.cpu()
        if (dynamic or dynamic_batch)
        else model,  # --(dynamic or dynamic_batch) only compatible with cpu
        im.cpu() if (dynamic or dynamic_batch) else im,
        f,
        verbose=False,
        opset_version=opset,
        # training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
        # do_constant_folding=not train,
        do_constant_folding=False,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {
        "stride": int(max(model.model.stride if end2end else model.stride)),
        "names": model.model.names if end2end else model.names,
    }
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(
                ("onnxruntime-gpu" if cuda else "onnxruntime", "onnx-simplifier>=0.4.1")
            )
            import onnxsim

            LOGGER.info(f"{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...")
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "assert check failed"
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f"{prefix} simplifier failure: {e}")

    if cleanup:
        try:
            print("\nStarting to cleanup ONNX using onnx_graphsurgeon...")
            import onnx_graphsurgeon as gs

            graph = gs.import_onnx(model_onnx)
            graph = graph.cleanup().toposort()
            model_onnx = gs.export_onnx(graph)
        except Exception as e:
            print(f"Cleanup failure: {e}")

    return f, model_onnx


@smart_inference_mode()
def run(
    data=ROOT / "data/coco128.yaml",  # 'dataset.yaml path'
    weights=ROOT / "yolov5s.pt",  # weights path
    imgsz=(640, 640),  # image (height, width)
    batch_size=1,  # batch size
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    include=("torchscript", "onnx"),  # include formats
    half=False,  # FP16 half-precision export
    inplace=False,  # set YOLOv5 Detect() inplace=True
    train=False,  # model.train() mode
    keras=False,  # use Keras
    optimize=False,  # TorchScript: optimize for mobile
    int8=False,  # CoreML/TF INT8 quantization
    dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
    simplify=False,  # ONNX: simplify model
    opset=12,  # ONNX: opset version
    verbose=False,  # TensorRT: verbose log
    workspace=4,  # TensorRT: workspace size (GB)
    nms=False,  # TF: add NMS to model
    agnostic_nms=False,  # TF: add agnostic NMS to model
    topk_per_class=100,  # TF.js NMS: topk per class to keep
    topk_all=100,  # TF.js NMS: topk for all classes to keep
    iou_thres=0.45,  # TF.js NMS: IoU threshold
    conf_thres=0.25,  # TF.js NMS: confidence threshold
    mask_resolution=56,
    pooler_scale=0.25,
    sampling_ratio=0,
    dynamic_batch=False,
    end2end=False,
    trt=False,
    cleanup=False,
    roi_align=False,
):
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()["Argument"][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(
        include
    ), f"ERROR: Invalid --include {include}, valid --include arguments are {fmts}"
    (
        jit,
        onnx,
        xml,
        engine,
        coreml,
        saved_model,
        pb,
        tflite,
        edgetpu,
        tfjs,
    ) = flags  # export booleans
    file = Path(
        url2file(weights) if str(weights).startswith(("http:/", "https:/")) else weights
    )  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    if half:
        assert (
            device.type != "cpu" or coreml
        ), "--half only compatible with GPU export, i.e. use --device 0"
        assert (
            not dynamic
        ), "--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both"
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model

    dynamic = False if dynamic_batch else dynamic
    dynamic = False if end2end else dynamic

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    if optimize:
        assert (
            device.type == "cpu"
        ), "--optimize not compatible with cuda devices, i.e. use --device cpu"

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    im = torch.zeros(batch_size, 3, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.train() if train else model.eval()  # training mode = no Detect() layer grid construction
    for k, m in model.named_modules():
        if isinstance(m, (Detect, IDetect)):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs
    if half and not coreml:
        im, model = im.half(), model.half()  # to FP16
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    LOGGER.info(
        f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)"
    )

    # Exports
    f = [""] * 10  # exported filenames
    warnings.filterwarnings(
        action="ignore", category=torch.jit.TracerWarning
    )  # suppress TracerWarning
    if jit:
        raise NotImplementedError
    if engine:  # TensorRT required before ONNX
        raise NotImplementedError
    if onnx or xml:  # OpenVINO requires ONNX
        f[2], _ = export_onnx(
            model=model,
            im=im,
            file=file,
            opset=opset,
            train=train,
            dynamic=dynamic,
            simplify=simplify,
            cleanup=cleanup,
            dynamic_batch=dynamic_batch,
            end2end=end2end,
            trt=trt,
            topk_all=topk_all,
            device=device,
            iou_thres=iou_thres,
            score_thres=conf_thres,
            mask_resolution=mask_resolution,
            pooler_scale=pooler_scale,
            sampling_ratio=sampling_ratio,
            image_size=imgsz,
            roi_align=roi_align,
        )
    if xml:  # OpenVINO
        raise NotImplementedError
    if coreml:
        raise NotImplementedError

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        h = "--half" if half else ""  # --half FP16 inference arg
        LOGGER.info(
            f"\nExport complete ({time.time() - t:.1f}s)"
            f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
            f"\nDetect:          python detect.py --weights {f[-1]} {h}"
            f"\nValidate:        python val.py --weights {f[-1]} {h}"
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')"
            f"\nVisualize:       https://netron.app"
        )
    return f  # return list of exported files/dirs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path"
    )
    parser.add_argument(
        "--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model.pt path(s)"
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="image (h, w)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="cpu", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--inplace", action="store_true", help="set YOLOv5 Detect() inplace=True")
    parser.add_argument("--train", action="store_true", help="model.train() mode")
    parser.add_argument("--keras", action="store_true", help="TF: use Keras")
    parser.add_argument("--optimize", action="store_true", help="TorchScript: optimize for mobile")
    parser.add_argument("--int8", action="store_true", help="CoreML/TF INT8 quantization")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--opset", type=int, default=12, help="ONNX: opset version")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT: workspace size (GB)")
    parser.add_argument("--nms", action="store_true", help="TF: add NMS to model")
    parser.add_argument("--agnostic-nms", action="store_true", help="TF: add agnostic NMS to model")
    parser.add_argument(
        "--topk-per-class", type=int, default=100, help="TF.js NMS: topk per class to keep"
    )
    parser.add_argument(
        "--topk-all", type=int, default=100, help="TF.js NMS: topk for all classes to keep"
    )
    parser.add_argument("--iou-thres", type=float, default=0.45, help="TF.js NMS: IoU threshold")
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="TF.js NMS: confidence threshold"
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=["onnx"],
        help="onnx",
    )
    parser.add_argument("--dynamic-batch", action="store_true", help="ONNX: dynamic batching")
    parser.add_argument("--end2end", action="store_true", help="ONNX: NMS")
    parser.add_argument("--trt", action="store_true", help="ONNX: TRT")
    parser.add_argument("--cleanup", action="store_true", help="ONNX: Cleanup")
    parser.add_argument(
        "--mask-resolution", type=int, default=56, help="ONNX: Roialign mask-resolution"
    )
    parser.add_argument(
        "--pooler-scale",
        type=float,
        default=0.25,
        help="ONNX: Roialign scale, scale = proto shape / input shape",
    )
    parser.add_argument(
        "--sampling-ratio", type=int, default=0, help="ONNX: Roialign sampling ratio"
    )
    parser.add_argument(
        "--roi-align", action="store_true", help="ONNX: Crop And Resize mask using roialign"
    )
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    for opt.weights in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
