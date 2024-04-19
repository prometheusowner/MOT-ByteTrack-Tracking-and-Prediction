import os
import torch
import yaml
from ultralytics import YOLO
from QtFusion.path import abs_path
device = "0" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    workers = 1
    batch = 8
    data_name = "PersonCar"
    data_path = abs_path(f'datasets/{data_name}/{data_name}.yaml', path_type='current')  # dataset yaml file absolute path
    unix_style_path = data_path.replace(os.sep, '/')

    # get path
    directory_path = os.path.dirname(unix_style_path)
    # Read YAML files in order
    with open(data_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    # edit the path
    if 'path' in data:
        data['path'] = directory_path
        # Writhe the edited data back to YAML file
        with open(data_path, 'w') as file:
            yaml.safe_dump(data, file, sort_keys=False)

    model = YOLO(abs_path('./weights/yolov8n.pt'), task='detect')  # Load pretrained model
    results2 = model.train(  # Start training
        data=data_path,  # the training data's path
        device=device,  # Use Gpu if CUDA available
        workers=workers,  # Specify using 1 worker processes to load data
        imgsz=640,  # Input image with 640x640
        epochs=100,  # training 100 epochs
        batch=batch,  # Specify the size of each batch to be 8
        name='train_v8_' + data_name  # set the name of training mission
    )

    model = YOLO(abs_path('./weights/yolov5nu.pt', path_type='current'), task='detect')  # 加载预训练的YOLOv8模型
    # model = YOLO('./weights/yolov5.yaml', task='detect').load('./weights/yolov5nu.pt')  # 加载预训练的YOLOv8模型
    # Training.
    results = model.train(  # 开始训练模型
        data=data_path,  # 指定训练数据的配置文件路径
        device=device,  # 自动选择进行训练
        workers=workers,  # 指定使用2个工作进程加载数据
        imgsz=640,  # 指定输入图像的大小为640x640
        epochs=100,  # 指定训练100个epoch
        batch=batch,  # 指定每个批次的大小为8
        name='train_v5_' + data_name  # 指定训练任务的名称
    )
