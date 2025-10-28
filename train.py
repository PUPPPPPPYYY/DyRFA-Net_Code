from ultralytics import YOLOv10


model_yaml_path = 'DyRFANet/YAML/DyRFA-Nat.yaml'
data_yaml_path = 'DyRFANet/data.yaml'
pre_model_name = 'DyRFANet/yolov10n.pt'

if __name__ == '__main__':
   
    model = YOLOv10(model_yaml_path).load(pre_model_name)
  
    results = model.train(data=data_yaml_path,
                          epochs=300,
                          batch=8
                          )

