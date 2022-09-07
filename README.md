# Export Yolov7-seg to ONNX and TensorRT

This implimentation is based on [yolov7](https://github.com/WongKinYiu/yolov7/tree/u7/seg).

## Install

## Usage

### Download model

### Export with roi-align

#### Export ONNX

- `python3 segment/export.py --data ./data/coco.yaml --weights ./weights/yolov7-seg.pt --batch-size 1 --device cpu --simplify --opset 14 --workspace 8 --iou-thres 0.65 --conf-thres 0.35 --include onnx --end2end --cleanup --dynamic-batch --roi-align --mask-resolution 96`

- [scripts](tools/Yolov7onnx_mask-roialign.ipynb)

#### Export TensorRT

- `python3 segment/export.py --data ./data/coco.yaml --weights ./weights/yolov7-seg.pt --batch-size 1 --device cpu --simplify --opset 14 --workspace 8 --iou-thres 0.65 --conf-thres 0.35 --include onnx --end2end --trt --cleanup --dynamic-batch --roi-align --mask-resolution 96`

- `CUDA_VISIBLE_DEVICES=1 /usr/src/tensorrt/bin/trtexec --onnx=./weights/yolov7-seg.onnx --saveEngine=./weights/yolov7-seg-nms.trt --workspace=8192 --fp16 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:8x3x640x640 --shapes=images:1x3x640x640`

- [scripts](tools/YOLOv7trt_mask-roialign.ipynb)

### Export without roi-align

#### Export ONNX

- `python3 segment/export.py --data ./data/coco.yaml --weights ./weights/yolov7-seg.pt --batch-size 1 --device cpu --simplify --opset 14 --workspace 8 --iou-thres 0.65 --conf-thres 0.35 --include onnx --end2end --cleanup --dynamic-batch`

- [scripts](tools/Yolov7onnx_mask.ipynb)

#### Export TensorRT

- `python3 segment/export.py --data ./data/coco.yaml --weights ./weights/yolov7-seg.pt --batch-size 1 --device cpu --simplify --opset 14 --workspace 8 --iou-thres 0.65 --conf-thres 0.35 --include onnx --end2end --trt --cleanup --dynamic-batch`

- `CUDA_VISIBLE_DEVICES=1 /usr/src/tensorrt/bin/trtexec --onnx=./weights/yolov7-seg.onnx --saveEngine=./weights/yolov7-seg-nms.trt --workspace=8192 --fp16 --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:8x3x640x640 --shapes=images:1x3x640x640`

- [scripts](tools/YOLOv7trt_mask.ipynb)
