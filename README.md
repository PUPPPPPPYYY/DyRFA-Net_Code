Our work is built upon YOLOv10n (YOLOv10: Real-Time End-to-End Object Detection), and the following sections provide a detailed explanation of the code.
## Installation
`conda` virtual environment is recommended. 
```
conda create -n DyRFA-Net python=3.9
conda activate DyRFA-Net
pip install -r requirements.txt
pip install -e .
```

## Validation
[`yolov10n.pt`](https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt) 
```
python val.py --weights best.pt --data data.yaml
 
```

## Training 
```
python train.py  --data data.yaml --epochs 300 --batch 8
```

## Export
```
# End-to-End ONNX
yolo export model=yolov10n/s/m/b/l/x.pt format=onnx opset=13 simplify
# Predict with ONNX
yolo predict model=yolov10n/s/m/b/l/x.onnx

# End-to-End TensorRT
yolo export model=yolov10n/s/m/b/l/x.pt format=engine half=True simplify opset=13 workspace=16
# Or
trtexec --onnx=yolov10n/s/m/b/l/x.onnx --saveEngine=yolov10n/s/m/b/l/x.engine --fp16
# Predict with TensorRT
yolo predict model=yolov10n/s/m/b/l/x.engine
```

## Acknowledgement

The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics) and [RT-DETR](https://github.com/lyuwenyu/RT-DETR).

Thanks for the great implementations! 

## Citation

Because our model is derived from and improves upon their framework, we acknowledge this by citing their paper.Thanks!
```BibTeX
@misc{wang2024yolov10,
      title={YOLOv10: Real-Time End-to-End Object Detection}, 
      author={Ao Wang and Hui Chen and Lihao Liu and Kai Chen and Zijia Lin and Jungong Han and Guiguang Ding},
      year={2024},
      eprint={2405.14458},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
