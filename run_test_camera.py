# -*- coding: utf-8 -*-
import sys  # 导入sys模块，用于处理Python运行时环境的一些操作
import time  # 导入time模块，用于处理时间相关的操作
import cv2  # 导入OpenCV库，用于处理图像和视频
import numpy as np
from QtFusion.path import abs_path
from QtFusion.config import QF_Config
from QtFusion.widgets import QMainWindow  # 从QtFusion库中导入FBaseWindow类，用于创建主窗口
from QtFusion.handlers import MediaHandler  # 从QtFusion库中导入MediaHandler类，用于处理媒体数据
from QtFusion.utils import drawRectBox  # 从QtFusion库中导入drawRectBox函数，用于在图像上绘制矩形框
from QtFusion.utils import get_cls_color  # 从QtFusion库中导入get_cls_color函数，用于获取类别颜色
from PySide6 import QtWidgets, QtCore  # 导入PySide6库的QtWidgets和QtCore模块，用于创建GUI和处理Qt的核心功能
from YOLOv8v5TrackModel import YOLOv8v5Tracker  # 从YOLOv8Model模块中导入YOLOv8Detector类，用于进行YOLOv8物体检测
from datasets.PersonCar.label_name import Label_list
from collections import defaultdict

QF_Config.set_verbose(False)
track_history = defaultdict(list)  # 初始化轨迹历史字典
retain_frames = 50


class MainWindow(QMainWindow):  # 定义MainWindow类，继承自FBaseWindow类
    def __init__(self):  # 定义构造函数
        super().__init__()  # 调用父类的构造函数
        self.resize(850, 500)  # 设置窗口的大小为850x500
        self.label = QtWidgets.QLabel(self)  # 创建一个QLabel对象，用于显示图像
        self.label.setGeometry(0, 0, 850, 500)  # 设置QLabel的位置和大小

    def keyPressEvent(self, event):  # 定义键盘按键事件处理函数
        if event.key() == QtCore.Qt.Key.Key_Q:  # 如果按下的是Q键
            self.close()  # 关闭窗口


def frame_process(image):
    global track_history  # 使用全局轨迹历史字典
    image = cv2.resize(image, (850, 500))  # 将图像的大小调整为850x500
    pre_img = model.preprocess(image)  # 对图像进行预处理

    t1 = time.time()  # 获取当前时间
    pred = model.predict(pre_img)  # 使用模型进行预测（这应当使用track方法）
    t2 = time.time()  # 获取当前时间
    use_time = t2 - t1  # 计算预测所花费的时间

    print("推理时间: %.2f" % use_time)  # 打印预测所花费的时间

    det = pred[0]  # 获取预测结果
    if det is not None and len(det):
        det_info = model.postprocess(pred)  # 对预测结果进行后处理
        for info in det_info:  # 遍历检测信息
            name, bbox, conf, cls_id, track_id = info['class_name'], info['bbox'], info['score'], info[
                'class_id'], info.get('track_id')
            if track_id is not None:
                center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))  # 计算边界框的中心点
                track_history[track_id].append(center)  # 更新轨迹历史
                if len(track_history[track_id]) > retain_frames:  # 保留最近retain_frames个轨迹点
                    track_history[track_id].pop(0)

                # 绘制轨迹线
                if len(track_history[track_id]) > 1:
                    track_color = colors[cls_id]  # 获取与标记框相同的颜色
                    cv2.polylines(image, [np.array(track_history[track_id], np.int32).reshape((-1, 1, 2))], False,
                                  track_color, 2)

            label = '%s %.0f%%' % (name, conf * 100)  # 创建标签，包含类别名称和置信度
            image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=colors[cls_id])  # 在图像上绘制边界框和标签

    window.dispImage(window.label, image)  # 在窗口的label上显示图像


if __name__ == '__main__':  # 如果当前模块是主模块
    cls_name = Label_list  # 定义类名列表

    model = YOLOv8v5Tracker()  # 创建YOLOv8Detector对象
    model.load_model(abs_path("weights/best-yolov8n.pt", path_type="current"))  # 加载预训练的YOLOv8模型
    colors = get_cls_color(model.names)  # 获取类别颜色

    app = QtWidgets.QApplication(sys.argv)  # 创建QApplication对象
    window = MainWindow()  # 创建MainWindow对象

    videoHandler = MediaHandler(fps=30)  # 创建MediaHandler对象，设置帧率为30
    videoHandler.frameReady.connect(frame_process)  # 当有新的帧准备好时，调用frame_process函数
    videoHandler.setDevice(device=0)  # 设置设备为0，即默认的摄像头
    videoHandler.startMedia()  # 开始处理媒体流

    # 显示窗口
    window.show()
    # 进入 Qt 应用程序的主循环
    sys.exit(app.exec())
