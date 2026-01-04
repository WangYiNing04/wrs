'''
Author: wang yining
Date: 2025-10-27 12:42:55
LastEditTime: 2025-10-27 13:26:58
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/yolo_train/train.py
Description: 
e-mail: wangyining0408@outlook.com
'''
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights


# Train the model
results = model.train(data="/home/wyn/Desktop/collected_images_water/collected_images_water/stack_bottles_three/YOLODataset/dataset.yaml", 
                      epochs=200,
                      imgsz=640,)
