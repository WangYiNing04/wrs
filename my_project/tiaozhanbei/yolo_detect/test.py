'''
Author: wang yining
Date: 2025-10-17 14:29:26
LastEditTime: 2025-10-17 14:29:34
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/yolo_detect/test.py
Description: 
e-mail: wangyining0408@outlook.com
'''
from ultralytics import YOLO

model = YOLO('./model/empty_cup_place/cup/best.pt')
print(model.model.names)  # 查看类别名称
print(model.model.args)   # 查看模型参数（是否有'keypoints'相关配置）