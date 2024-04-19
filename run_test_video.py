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
        # 添加一个用于显示统计信息的标签
        self.stats_label = QtWidgets.QLabel(self)
        self.stats_label.setGeometry(0, 450, 850, 50)  # 设置统计信息标签的位置和大小
        self.stats_label.setAlignment(QtCore.Qt.AlignCenter)  # 设置文本居中显示
        # 设置背景色和反色的文本颜色
        self.setLabelBackgroundAndTextColor(self.stats_label, (0, 0, 0))  # 假设背景色为黑色

    def setLabelBackgroundAndTextColor(self, label, background_rgb):
        # 计算反色
        inverse_rgb = tuple(255 - x for x in background_rgb)
        # 设置样式表
        label.setStyleSheet(f"background-color: rgb{background_rgb};"
                            f"color: rgb{inverse_rgb};")
    def keyPressEvent(self, event):  # 定义键盘按键事件处理函数
        if event.key() == QtCore.Qt.Key.Key_Q:  # 如果按下的是Q键
            self.close()  # 关闭窗口
            out_video.release()  # 在关闭窗口的时候释放视频写入资源



def frame_process(image):
    global track_history  # 使用全局轨迹历史字典
    image = cv2.resize(image, (850, 500))  # 将图像的大小调整为850x500
    pre_img = model.preprocess(image)  # 对图像进行预处理

    t1 = time.time()  # 获取当前时间
    pred = model.predict(pre_img)  # 使用模型进行预测（这应当使用track方法）
    t2 = time.time()  # 获取当前时间
    use_time = t2 - t1  # 计算预测所花费的时间

    print("Time Used for detection: %.2f" % use_time)  # 打印预测所花费的时间
    print("Time Used for detection: %.2f" % use_time)  # 打印预测所花费的时间
    det = pred[0]  # 获取预测结果
    class_counts = {cls_name: 0 for cls_name in cls_name}# 初始化类别计数字典
    if det is not None and len(det):
        det_info = model.postprocess(pred)  # 对预测结果进行后处理
        for info in det_info:  # 遍历检测信息
            name, bbox, conf, cls_id, track_id = info['class_name'], info['bbox'], info['score'], info[
                'class_id'], info.get('track_id')
            class_name = model.names[cls_id]
            label = f'{name} {conf * 100:.0f}%'  # Define label with class name and confidence percentage
            if track_id is not None:
                center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))  # 计算边界框的中心点
                track_history[track_id].append(center)  # 更新轨迹历史
                if len(track_history[track_id]) > retain_frames:  # 保留最近retain_frames个轨迹点
                    track_history[track_id].pop(0)

                # 绘制轨迹线
                if len(track_history[track_id]) > 1:
                    track_color = colors[cls_id]  # 获取与标记框相同的颜色
                    cv2.polylines(image, [np.array(track_history[track_id], np.int32).reshape((-1, 1, 2))], False, track_color, 2)
                    # 计算当前帧和前一帧之间的位移差
                    dx = center[0] - track_history[track_id][-2][0]
                    dy = center[1] - track_history[track_id][-2][1]

                    # 计算速率（单位：像素/帧），假设视频帧率为 frame_rate
                    velocity_x = dx
                    velocity_y = dy
                    # 输出类别和速率
                    print(f"{class_name} (ID: {track_id}): Velocity x = {velocity_x} pixels/frame, y = {velocity_y} pixels/frame, Label: {label}")

            label = '%s %.0f%%' % (name, conf * 100)  # 创建标签，包含类别名称和置信度
            image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=colors[cls_id])  # 在图像上绘制边界框和标签
            # 计数每个类别的个数
            class_counts[name] += 1

    # 构建并显示统计信息
    stats_text = "Counts: " + ", ".join([f"{k}: {v}" for k, v in class_counts.items()])
    window.stats_label.setText(stats_text)  # 更新统计信息标签的文本

    window.dispImage(window.label, image)  # 在窗口的label上显示图像
    # 将处理后的帧写入输出视频
    out_video.write(image)


if __name__ == '__main__':  # 如果当前模块是主模块
    cls_name = Label_list  # 定义类名列表
    last_track_info = 0
    model = YOLOv8v5Tracker()  # 创建YOLOv8Detector对象
    model.load_model(abs_path("E:/Python_Projects/YOLO_Bytetrack/runs/detect/train_v5_PersonCar/weights/best.pt", path_type="current"))  # 加载预训练的YOLOv8模型
    colors = get_cls_color(model.names)  # 获取类别颜色

    out_video_path = abs_path("output_video.avi", path_type="current")  # 定义输出视频的路径
    # 使用H.264编码器和MP4格式
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 也可以尝试使用 'avc1'
    out_video = cv2.VideoWriter(out_video_path, fourcc, 30.0, (850, 500))

    app = QtWidgets.QApplication(sys.argv)  # 创建QApplication对象
    window = MainWindow()  # 创建MainWindow对象

    filename = abs_path("test_media/JapanStreet.mp4", path_type="current")  # 定义视频文件的路径
    videoHandler = MediaHandler(fps=30)  # 创建MediaHandler对象，设置帧率为30fps
    videoHandler.frameReady.connect(frame_process)  # 当有新的帧准备好时，调用frame_process函数进行处理
    videoHandler.setDevice(filename)  # 设置视频源
    videoHandler.startMedia()  # 开始处理媒体

    # 显示窗口
    window.show()

    # 进入 Qt 应用程序的主循环
    sys.exit(app.exec())
    out_video.release()