此目录用于存放YOLO模型文件

需要下载以下文件到此目录:

1. yolov3.cfg - YOLO配置文件
   下载地址: https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg

2. yolov3.weights - YOLO预训练权重
   下载地址: https://pjreddie.com/media/files/yolov3.weights

3. coco.names - COCO数据集的类别名称
   下载地址: https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

您可以使用以下命令下载这些文件:

```bash
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg
wget https://pjreddie.com/media/files/yolov3.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```

这些文件将被系统用于对象检测。如果您想使用其他YOLO模型版本，请相应地修改launch文件中的配置参数。 