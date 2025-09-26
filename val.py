import os
from ultralytics import YOLOv10
# 模型加载
model = YOLOv10("DyRFA-Net/runs/detect/train/weights/best.pt")
# 数据配置文件路径
data_config_path = "DyRFA-Net/data.yaml"
# 进行验证并输出评估结果
results = model.val(data=data_config_path,save_json=True)  # `save_json` 参数用于保存详细评估结果
# 打印评估结果
if hasattr(results, 'metrics'):
    metrics = results.metrics
    print(f"Class  Images  Instances  Box(P)  R  mAP50  mAP75  mAP90  mAP50-95")
    for class_name, class_metrics in metrics.items():
        print(f"{class_name}  {class_metrics['images']}  {class_metrics['instances']}  {class_metrics['Box(P)']}  {class_metrics['R']}  {class_metrics['mAP50']}  {class_metrics['mAP75']}  {class_metrics['mAP95']}  {class_metrics['mAP50-95']}")
else:
    print("No detailed metrics found in the results.")
