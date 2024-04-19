# -*- coding: utf-8 -*-
import os
import time
from collections import defaultdict
from os import getcwd
import cv2
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QFileDialog, QMessageBox
from QtFusion.handlers import MediaHandler, ImageHandler
from QtFusion.path import abs_path, get_script_dir
from QtFusion.utils import cv_imread, drawRectEdge, drawRectBox
from QtFusion.widgets import QMainWindow, QWindowCtrls, updateTable

from Recognition_UI import Ui_MainWindow
from YOLOv8v5TrackModel import YOLOv8v5Tracker, count_classes
from datasets.PersonCar.label_name import Label_list


class RecMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None, *args, **kwargs):
        super(RecMainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)  # 从Ui_MainWindow中导入界面生成
        self.velocity_history = defaultdict(list)  # 存储每个跟踪ID的速度历史
        # 设置界面最小化、退出等控制按钮及弹窗参数
        QWindowCtrls(self, exit_title="Multi-tracking detector",  # 退出确认框标题
                     exit_message='Done？',  # 退出确认框text
                     button_sizes=(20, 20),  # 窗口控制按钮的大小
                     button_gaps=30,  # 窗口控制按钮间的间距
                     button_right_margin=50,  # 窗口控制按钮距离右侧的间距
                     hint_flag=False)  # 是否隐藏退出确认框的边框

        self.setUiStyle(windowFlag=True, transBackFlag=True)  # 设置界面样式: 无边框/透明

        self.yaml_file = abs_path("themes/Settings_main.yaml", path_type="current")  # 以当前文件为基础的绝对路径
        self.loadYamlSettings(yaml_file=self.yaml_file, base_path=get_script_dir())  # 设置图标背景或文字，使用当前文件夹为基础路径
        qss_path = abs_path("themes/main_dark_back.qss", path_type="current")
        self.loadStyleSheet(qssFilePath=qss_path)  # 设置样式表
        self.start_time = time.time()
        self.file_path = getcwd()  # 获取当前工作目录

        # 创建 MediaHandler 实例，用于处理摄像头、视频和图像
        self.CAM_NUM = 0  # 默认摄像头标号
        self.cameraHandler = MediaHandler(device=self.CAM_NUM, fps=30)  # 摄像头处理器，帧率30
        self.videoHandler = MediaHandler(fps=30)  # 视频处理器，帧率30
        self.imageHandler = ImageHandler()  # 图片处理器

        self.detInfo = []  # 存储检测结果的列表
        self.current_image = []  # 当前处理的图像
        self.saved_images = []  # 待保存的图像序列
        self.detected_image = None  # 检测结果图像
        self.track_history = defaultdict(list)  # 用于保存轨迹线
        self.track_lens = 30  # 用于轨迹线的长度

        self.slot_init()  # 定义槽函数

        self.id_tab = 0  # 表格行数，用于记录识别识别条目
        self.count_name = Label_list  # 定义类名列表
        self.count_table = []  # 存储花卉识别计数的列表
        self.colors = self.get_cls_color(self.count_name)  # 根据类别产生对应标记的颜色

        self.Sidebar.setFixedWidth(55)  # 默认隐藏侧边栏
        self.animation = QtCore.QPropertyAnimation(self.Sidebar, b"minimumWidth")  # 设置动画效果
        self.animation.setDuration(400)

        self.pass_flag = False  # 标记是否通过
        self.total_frames = 1000
        self.cur_frames = 0

        # 加载模型
        self.model = YOLOv8v5Tracker()  # 创建YOLOv8Detector模型实例
        # 加载训练的模型权重
        self.model_path = abs_path("E:/Python_Projects/YOLO_Bytetrack/runs/detect/train_v5_PersonCar/weights/best.pt",
                                   path_type="current")
        self.model.load_model(model_path=self.model_path)
        self.colors = self.get_cls_color(self.model.names)
        self.plot_vertical_bar(self.label_bar, self.count_name, [0 for i in self.count_name], self.colors, margin=10)
        self.ratio = 1

    def showTime(self):
        """
        显示主窗口。
        """
        self.show()

    def setupUi(self, MainWindow):
        """
        设置主窗口的UI。
        """
        Ui_MainWindow.setupUi(self, MainWindow)
        self.tableWidget.setColumnWidth(0, 80)  # 设置表格列宽
        self.tableWidget.setColumnWidth(1, 200)
        self.tableWidget.setColumnWidth(2, 150)
        self.tableWidget.setColumnWidth(3, 200)
        self.tableWidget.setColumnWidth(4, 120)

    def clearUI(self):
        """
        清除主窗口的UI，准备进行新一轮的识别。
        """
        self.loadYamlSettings(yaml_file=self.yaml_file, base_path=get_script_dir())  # 设置图标背景或文字，使用当前文件夹为基础路径
        self.label_display.clear()  # 清除显示标签上的内容
        self.model.load_model(self.model_path)
        self.track_history = defaultdict(list)
        QtWidgets.QApplication.processEvents()

    def slot_init(self):
        """
        初始化槽函数，用于连接各个信号和槽函数。
        """
        # 连接摄像头相关的槽函数
        self.cameraHandler.frameReady.connect(self.frame_process)  # 摄像头每帧数据准备时触发frame_process槽函数
        self.cameraHandler.mediaOpened.connect(self.handle_camera_opened)  # 摄像头成功打开时触发handle_camera_opened槽函数
        self.cameraHandler.mediaFailed.connect(self.handle_camera_failed)  # 摄像头打开失败时触发handle_camera_failed槽函数
        self.cameraHandler.mediaClosed.connect(self.handle_closed)  # 摄像头关闭时触发handle_closed槽函数
        self.cameraHandler.stopOtherActivities.connect(self.stopActivities)  # 启动摄像头时停止其他活动
        self.toolButton_camera.clicked.connect(self.toggle_camera)  # 点击摄像头按钮时触发toggle_camera槽函数

        # 连接视频相关的槽函数
        self.videoHandler.frameReady.connect(self.frame_process)  # 视频每帧数据准备时触发frame_process槽函数
        self.videoHandler.mediaOpened.connect(self.handle_video_opened)  # 视频成功打开时触发handle_video_opened槽函数
        self.videoHandler.mediaClosed.connect(self.handle_closed)  # 视频关闭时触发handle_closed槽函数
        self.videoHandler.mediaFailed.connect(self.handle_video_failed)  # 视频打开失败时触发handle_video_failed槽函数
        self.videoHandler.stopOtherActivities.connect(self.stopActivities)  # 启动视频时停止其他活动
        self.toolButton_video.clicked.connect(self.toggle_video)  # 点击视频按钮时触发toggle_video槽函数

        # 连接图像相关的槽函数
        self.imageHandler.frameReady.connect(self.frame_process)  # 图像每帧数据准备时触发frame_process槽函数
        self.imageHandler.imageOpened.connect(self.handle_image_opened)  # 图像成功打开时触发handle_image_opened槽函数
        self.imageHandler.imageClosed.connect(self.handle_closed)  # 图像关闭时触发handle_closed槽函数
        self.imageHandler.imageFailed.connect(self.handle_image_failed)  # 图像打开失败时触发handle_image_failed槽函数
        self.imageHandler.stopOtherActivities.connect(self.stopActivities)  # 启动图像处理时停止其他活动
        self.toolButton_file.clicked.connect(self.toggle_image)  # 点击选择图片按钮时触发toggle_image槽函数
        self.toolButton_folder.clicked.connect(self.toggle_folder)  # 点击选择文件夹按钮时触发toggle_folder槽函数

        # 连接其他功能的槽函数
        self.toolButton_model.clicked.connect(self.toggle_model)  # 点击选择模型按钮时触发toggle_model槽函数
        self.comboBox_select.currentIndexChanged.connect(self.toggle_comboBox)  # 下拉框选择变化时触发toggle_comboBox槽函数
        self.tableWidget.cellPressed.connect(self.toggle_table_review)  # 单元格被按下时触发toggle_table_review槽函数
        self.toolButton_saveing.clicked.connect(self.toggle_saveFile)  # 点击保存文件按钮时触发toggle_saveFile槽函数
        self.toolButton_menu.clicked.connect(self.toggle_settings_drawer)  # 点击菜单按钮时触发toggle_settings_drawer槽函数
        self.pushButton_hide.clicked.connect(self.toggle_settings_drawer)  # 点击隐藏按钮时触发toggle_settings_drawer槽函数

        self.slider_conf.valueChanged.connect(self.update_tooltip)  # 滑块触发槽函数
        self.slider_iou.valueChanged.connect(self.update_tooltip)  # 滑块触发槽函数

    def toggle_settings_drawer(self):
        """
        当点击按钮时，如果侧边栏被隐藏，则显示；如果侧边栏已显示，则隐藏。
        """
        if self.Sidebar.width() == 55:  # 检查侧边栏的宽度是否为55
            self.animation.setStartValue(55)  # 设置动画起始值
            self.animation.setEndValue(240)  # 设置动画结束值，显示侧边栏
            self.animation.start()  # 启动动画
        else:
            self.animation.setStartValue(240)  # 设置动画起始值
            self.animation.setEndValue(55)  # 设置动画结束值，隐藏侧边栏
            self.animation.start()  # 启动动画

    def toggle_table_review(self, row, col):
        """
        当点击表格中的单元格时，执行以下操作：
        - 读取所选行的文件路径、识别结果和坐标
        - 根据文件路径读取图片，并调整大小
        - 绘制识别结果的矩形框和标签在图片上
        - 在界面中显示图片和结果信息
        """
        try:
            if col == 0:  # 只在点击第一列时执行操作
                this_path = self.tableWidget.item(row, 1)  # 获取所选行的文件路径
                res = self.tableWidget.item(row, 2)  # 获取所选行的识别结果
                axes = self.tableWidget.item(row, 3)  # 获取所选行的坐标

                if (this_path is not None) & (res is not None) & (axes is not None):
                    this_path = this_path.text()  # 获取文件路径的文本
                    if os.path.isfile(this_path):  # 检查文件路径是否存在
                        res = res.text()  # 获取识别结果的文本
                        axes = axes.text()  # 获取坐标的文本

                        image = cv_imread(this_path)  # 读取选择的图片
                        # 获取原始视频帧的尺寸
                        height, width = image.shape[:2]

                        # 如果需要调整大小，保持原始宽高比
                        if width != 640 or height != 640:
                            # 你可以选择保持视频的原始尺寸，或者调整到特定的大小
                            # 只要保持宽高比即可
                            # 例如，将宽度调整到640，同时保持宽高比
                            ratio = 640.0 / width
                            new_width = 640
                            new_height = int(height * ratio)
                            image = cv2.resize(image, (new_width, new_height))

                        axes = [int(i) for i in axes.split(",")]  # 将坐标转换为整数列表
                        confi = float(self.tableWidget.item(row, 4).text())  # 获取置信度值

                        count = self.count_table[row]  # 获取花卉识别计数列表中的值
                        self.plot_vertical_bar(self.label_bar, self.count_name, count, self.colors, margin=10)
                        self.label_numer_result.setText(str(sum(count)))  # 在界面中显示总计数

                        image = drawRectEdge(image, axes, alpha=0.2, addText=res)  # 绘制矩形框和标签在图片上
                        self.dispImage(self.label_display, image)  # 在界面中显示图片

                        # 在界面标签中显示结果
                        self.label_xmin_result.setText(str(int(axes[0])))  # 标记框坐标显示
                        self.label_ymin_result.setText(str(int(axes[1])))
                        self.label_xmax_result.setText(str(int(axes[2])))
                        self.label_ymax_result.setText(str(int(axes[3])))
                        self.label_score_result.setText(str(round(confi * 100, 2)) + "%")  # 置信度值显示
                        self.label_class_result.setText(res)  # 类别显示
                        QtWidgets.QApplication.processEvents()
        except:
            self.label_display.setText('check the record！')

    def toggle_comboBox(self):
        """
        当选择下拉框中的不同项时，根据选项索引执行以下操作：
        - 根据选项索引更新界面中的类别、置信度、位置坐标和绘制框的样式
        """
        QtWidgets.QApplication.processEvents()
        image = self.current_image.copy()  # 复制当前处理的图像

        ind = self.comboBox_select.currentIndex() - 1  # 获取选项索引，并减去1（因为第一项是"全部"）
        ind_select = ind
        if ind <= -1:
            ind_select = 0

        if len(self.detInfo) > 0:  # 检查检测结果列表是否为空
            self.label_class_result.setText(self.detInfo[ind_select]['class_name'])  # 显示类别
            self.label_score_result.setText('%.2f' % (self.detInfo[ind_select]['score']))  # 显示置信度值
            # 显示位置坐标
            self.label_xmin_result.setText(str(int(self.detInfo[ind_select]['bbox'][0])))
            self.label_ymin_result.setText(str(int(self.detInfo[ind_select]['bbox'][1])))
            self.label_xmax_result.setText(str(int(self.detInfo[ind_select]['bbox'][2])))
            self.label_ymax_result.setText(str(int(self.detInfo[ind_select]['bbox'][3])))

            for i, det in enumerate(self.detInfo):  # 遍历所有标记框
                if ind != -1:
                    if ind != i:
                        continue
                # 在图像上标记目标框
                name, bbox, conf, cls_id = det['class_name'], det['bbox'], det['score'], det['class_id']
                label = '%s %.0f%%' % (name, conf * 100)  # 标记框左上方的类别显示
                self.label_score_result.setText(str('%.2f' % conf))  # 在界面上显示置信度值
                image = drawRectEdge(image, bbox, alpha=0.25, addText=label, color=self.colors[cls_id])  # 在图上添加标记框
            # 在Qt界面中显示检测完成画面
            self.dispImage(self.label_display, image)

    def toggle_model(self):
        """
        当点击模型按钮时，执行以下操作：
        - 停止所有活动
        - 清除UI上的标签显示
        - 弹出文件选择对话框，选择模型文件
        - 根据选择的模型文件加载模型并更新颜色标记
        """
        self.stopActivities()  # 停止所有活动
        self.clearUI()  # 清除UI上的标签显示

        filename, filetype = QFileDialog.getOpenFileName(self.centralwidget,
                                                         "Select model", getcwd(),  # 起始路径
                                                         "Model File (*.pt)")  # 文件类型

        if filename != '':
            self.textEdit_model.setText(filename + ' selected')
            self.toolButton_model.setToolTip(filename + 'selected')
            self.model_path = filename
            self.model.load_model(self.model_path)
            self.colors = self.get_cls_color(self.model.names)  # 根据类别产生对应标记的颜色
        else:
            self.textEdit_model.setText('Select model')
            self.toolButton_model.setToolTip('use default model')

    def toggle_image(self):
        """
        当点击图像按钮时，执行以下操作：
        - 停止图像处理器的活动
        - 弹出文件选择对话框，选择图像文件
        - 设置图像处理器的路径并启动图像处理
        """
        self.imageHandler.stopProcess()  # 停止图像处理器的活动

        filename, filetype = QFileDialog.getOpenFileName(self.centralwidget, "select file",
                                                         self.file_path,  # 起始路径
                                                         "Image(*.jpg;*.jpeg;*.png)")  # 文件类型

        self.file_path = filename  # 保存路径
        if filename:
            self.total_frames = 1
            self.saved_images = []  # 重置待保存图像序列
            self.imageHandler.setPath(filename)  # 设置图像处理器的路径
            self.imageHandler.startProcess()  # 启动图像处理
        else:
            self.imageHandler.stopProcess()  # 停止图像处理

    def toggle_folder(self):
        """
        当点击文件夹按钮时，执行以下操作：
        - 停止图像处理器的活动
        - 弹出文件选择对话框，选择文件夹
        - 设置图像处理器的路径并启动图像处理
        """
        self.imageHandler.stopProcess()  # 停止图像处理器的活动

        dir_choose = QFileDialog.getExistingDirectory(self.centralwidget, "select folder", self.file_path)

        self.file_path = dir_choose  # 保存路径
        if dir_choose:
            self.total_frames = 1
            self.saved_images = []  # 重置待保存图像序列
            self.imageHandler.setPath(dir_choose)  # 设置图像处理器的路径
            self.imageHandler.startProcess()  # 启动图像处理
        else:
            self.imageHandler.stopProcess()

    def stopActivities(self):
        """
        停止所有活动：
        - 如果视频处理器处于活动状态，停止媒体播放
        - 如果摄像头处理器处于活动状态，停止媒体播放
        - 如果图像处理器处于活动状态，停止图像处理
        """
        if self.videoHandler.isActive():
            self.videoHandler.stopMedia()  # 停止视频处理器的媒体播放
        if self.cameraHandler.isActive():
            self.cameraHandler.stopMedia()  # 停止摄像头处理器的媒体播放
        if self.imageHandler.isActive():
            self.imageHandler.stopProcess()  # 停止图像处理器的活动

    def handle_image_opened(self):
        """
        处理图像打开事件：
        - 清除UI上的内容，准备运行识别程序
        - 在UI上绘制初始的垂直条形图
        - 检查文件路径是文件还是文件夹，并在相应的文本框中显示路径信息
        - 在标签中显示启动识别系统的提示信息
        """
        self.clearUI()  # 清除UI上的内容

        if os.path.isfile(self.file_path):  # 检查文件路径是否为文件
            self.textEdit_image.setText(self.file_path + ' selected')
        elif os.path.isdir(self.file_path):  # 检查文件路径是否为文件夹
            self.textEdit_imgFolder.setText(self.file_path + ' selected')
        else:
            print('not acceptable path')
        # self.label_display.setText('正在启动识别系统...\n\nleading')  # 在标签中显示启动识别系统的提示信息

    def handle_image_failed(self, error):
        """
        处理图像打开失败事件：
        - 在控制台打印错误信息
        """
        print("Failed to open image file: {}".format(error))
        # self.label_display.setText("Failed to open image file: {}".format(error))

    def toggle_video(self):
        """
        当点击视频按钮时，执行以下操作：
        - 停止视频处理器的媒体播放
        - 弹出文件选择对话框，选择视频文件
        - 设置视频处理器的设备为选中的视频文件并启动媒体播放
        """
        self.videoHandler.stopMedia()  # 停止视频处理器的媒体播放

        filename, filetype = QFileDialog.getOpenFileName(self, "select",
                                                         "",  # 起始路径
                                                         "video(*.mp4;*.avi)")  # 文件类型

        if filename:
            self.file_path = filename  # 更新文件路径
            self.videoHandler.setDevice(filename)  # 设置视频处理器的设备为选中的视频文件
            self.videoHandler.startMedia()  # 启动视频处理器的媒体播放

            Info = self.videoHandler.getMediaInfo()
            self.total_frames = Info.get("frames", 1000)  # 视频总帧数
            self.cur_frames = 0  # 当前帧
            self.progressBar.setValue(0)
            self.saved_images = []  # 重置待保存图像序列
            # print(Info)
        else:
            self.videoHandler.stopMedia()

    def handle_video_opened(self):
        """
        处理视频打开事件：
        - 清理UI，准备运行识别程序
        - 在UI上绘制初始的垂直条形图
        - 在视频文本框中显示选中的视频文件路径
        - 在标签中显示启动识别系统的提示信息
        """
        self.clearUI()  # 清除UI上的内容
        # 绘制初始的垂直条形图
        self.textEdit_video.setText(self.file_path + ' selected')  # 在视频文本框中显示选中的视频文件路径
        self.label_display.setText('Starting program...\n\nleading')  # 在标签中显示启动识别系统的提示信息

    def handle_video_failed(self, error):
        """
        处理视频打开失败事件：
        - 在控制台打印错误信息
        """
        print("Failed to open video file: {}".format(error))
        # self.label_display.setText("Failed to open video file: {}".format(error))

    def toggle_camera(self):
        """
        当点击摄像头按钮时，执行以下操作：
        - 根据当前摄像头标号更新文件路径
        - 如果摄像头处理器处于非活动状态，则打开摄像头并启动定时器
        - 如果摄像头处理器处于活动状态，则关闭摄像头并停止定时器
        """
        self.file_path = 'Camera ' + str(self.CAM_NUM)  # 根据当前摄像头标号更新文件路径

        self.total_frames = 1000
        self.total_frames = 1000
        if not self.cameraHandler.isActive():  # 检查摄像头处理器的定时器状态
            self.cameraHandler.startMedia()  # 打开摄像头并启动定时器
            self.saved_images = []  # 重置待保存图像序列
        else:
            self.cameraHandler.stopMedia()  # 关闭摄像头并停止定时器

    def handle_camera_opened(self):
        """
        处理摄像头打开事件：
        - 清理UI，准备运行识别程序
        - 在UI上绘制初始的垂直条形图
        - 在摄像头文本框中显示摄像头已启动的信息
        - 在标签中显示启动识别系统的提示信息
        - 更新界面
        """
        self.clearUI()  # 清理UI上的内容
        # 绘制初始的垂直条形图
        self.textEdit_camera.setText('Camera On!')  # 在摄像头文本框中显示摄像头已启动的信息
        self.label_display.setText('Starting program...\n\nleading')  # 在标签中显示启动识别系统的提示信息
        QtWidgets.QApplication.processEvents()  # 更新界面

    def handle_camera_failed(self, error_message):
        """
        处理摄像头打开失败事件：
        - 弹出警告对话框，显示错误信息
        """
        QMessageBox.warning(self, "Warning", "Check your camera！\n " + error_message,
                            buttons=QMessageBox.Ok, defaultButton=QMessageBox.Ok)

    def handle_closed(self):
        """
        处理关闭事件：
        - 清理UI上的内容
        - 清空下拉框并添加"所有目标"选项
        - 清空标签的显示文字
        """
        self.clearUI()  # 清理UI上的内容
        self.comboBox_select.clear()  # 清空下拉框中的选项
        self.comboBox_select.addItem('All target')  # 添加"所有目标"选项
        self.label_display.setText('')  # 清空标签的显示文字
        self.total_frames = 1000  # 重置默认视频总帧数
        self.cur_frames = 0  # 重置当前帧
        self.progressBar.setValue(0)  # 重置进度条
        self.track_history = defaultdict(list)

    def frame_process(self, image):
        """
        处理每一帧图像的函数：
        - 显示摄像头画面
        - 对图像进行预处理
        - 使用模型进行预测
        - 处理预测结果并更新界面显示
        """
        # 显示摄像头画面
        # self.label_display.clear()
        # pre_img = image.copy()

        # 获取原始视频帧的尺寸
        height, width = image.shape[:2]

        # 如果需要调整大小，保持原始宽高比
        if width != 640 or height != 640:
            # 你可以选择保持视频的原始尺寸，或者调整到特定的大小
            # 只要保持宽高比即可
            # 例如，将宽度调整到640，同时保持宽高比
            self.ratio = 640.0 / width
            new_width = 640
            new_height = int(height * self.ratio)
            image = cv2.resize(image, (new_width, new_height))

        self.current_image = image.copy()  # 保存当前图像副本

        pre_img = self.model.preprocess(image)  # 对图像进行预处理

        # 更新模型参数
        params = {'conf': (self.slider_conf.value() + 1) / 100, 'iou': (self.slider_iou.value() + 1) / 100}
        self.model.set_param(params)

        t1 = time.time()
        pred = self.model.predict(pre_img)  # 使用模型进行预测
        t2 = time.time()
        use_time = t2 - t1  # 单张图片推理时间
        elapsed_time = t2 - self.start_time  # 计算从视频开始到当前帧的时间
        self.label_time_result.setText(str(round(use_time, 2)))  # 将推理时间放到右上角

        det = pred[0]  # 获取预测结果

        # 如果有检测信息则进行处理
        if det is not None and len(det):
            det_info = self.model.postprocess(pred)  # 后处理预测结果
            self.detInfo = det_info.copy()  # 保存检测信息
            if len(det_info):
                count = count_classes(det_info, self.count_name)  # 统计各个类别的数目
                for _ in det_info:
                    self.count_table.append(count)  # 记录各个类别数目

                self.label_numer_result.setText(str(sum(count)))  # 更新界面上的目标总数

                name, bbox, conf, cls_id = det_info[0]['class_name'], det_info[0]['bbox'], \
                    det_info[0]['score'], det_info[0]['class_id']
                self.label_class_result.setText(name)  # 显示类别
                self.label_score_result.setText('%.2f' % conf)  # 显示置信度
                self.label_xmin_result.setText(str(bbox[0]))  # 显示边界框的坐标
                self.label_ymin_result.setText(str(bbox[1]))
                self.label_xmax_result.setText(str(bbox[2]))
                self.label_ymax_result.setText(str(bbox[3]))

                # 更新下拉框
                self.comboBox_select.currentIndexChanged.disconnect(self.toggle_comboBox)  # 断开下拉框的信号连接
                self.comboBox_select.clear()  # 清空下拉框
                self.comboBox_select.addItem('All target')  # 添加"所有目标"选项

                self.plot_vertical_bar(self.label_bar, self.count_name, count, self.colors, margin=10)

                for i in range(len(det_info)):
                    text = "{}-{}".format(det_info[i]['class_name'], i + 1)
                    self.comboBox_select.addItem(text)  # 添加每个检测结果的选项
                self.comboBox_select.currentIndexChanged.connect(self.toggle_comboBox)  # 重新连接下拉框的信号

                for info in det_info:  # 遍历检测信息
                    track_id = info.get('track_id')
                    name, bbox, conf, cls_id = info['class_name'], info['bbox'], info['score'], info['class_id']
                    track_color = self.colors[cls_id]  # 默认颜色基于类别ID

                    if track_id is not None:
                        bbox = info['bbox']
                        center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))

                        # 限制轨迹长度
                        if len(self.track_history[track_id]) > self.track_lens:
                            self.track_history[track_id].pop(0)

                        # 绘制轨迹线
                        if len(self.track_history[track_id]) > 1:
                            points = np.array(self.track_history[track_id], np.int32).reshape((-1, 1, 2))
                            cv2.polylines(image, [points], False, track_color, 3)
                            # 计算当前帧和前一帧之间的位移差
                            dx = center[0] - self.track_history[track_id][-1][0]
                            dy = center[1] - self.track_history[track_id][-1][1]
                            cls = info['class_name']  # 获取类别名称
                            self.velocity_history[track_id].append((dx, dy, cls, elapsed_time))  # 更新速度历史，包括类别
                            print(f"Velocity for {track_id}: dx={dx}, dy={dy}, class={cls}, time={elapsed_time:.2f} seconds")  # 打印速度和类别来监控数据
                        self.track_history[track_id].append(center)

                        # 为跟踪的对象绘制边界框和标签
                        # label = '%s:%d %.0f%%' % (name, track_id if track_id is not None else 0, conf * 100)
                        label = '{}-{}'.format(name, track_id if track_id is not None else 0)
                        image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=track_color)

                        # 更新表格
                        updateTable(self.tableWidget, self.id_tab, self.file_path, name, bbox, '%.2f' % conf)
                        self.id_tab += 1

        self.dispImage(self.label_display, image)  # 在界面上显示图像
        self.detected_image = image  # 保存检测后的图像
        self.saved_images.append(image)  # 保存图像序列
        self.cur_frames = self.cur_frames + 1 if self.cur_frames + 1 <= self.total_frames else 0
        self.progressBar.setValue(self.cur_frames / self.total_frames * 100)
        QtWidgets.QApplication.processEvents()  # 更新界面

    def toggle_saveFile(self):
        """
        当点击保存文件按钮时，执行以下操作：
        - 检查是否有检测后的图像可供保存
        - 获取当前时间并生成保存文件名
        - 使用OpenCV将图像保存为PNG文件和视频文件
        - 在消息框中显示保存成功或失败的提示信息
        """
        if self.saved_images:  # 检查列表是否不为空
            # 显示开始保存的消息
            QMessageBox.information(self.centralwidget, "Saving the file", "Press OK\nstarting saving")

            # 执行保存操作
            now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))  # 获取当前时间
            if len(self.saved_images) == 1:
                # 只有一张图像时，保存为图片
                cv2.imwrite('./pic_' + str(now_time) + '.png', self.saved_images[0])
                QMessageBox.information(self.centralwidget, "Save file", "\nSuccessed!\nIt is saved！")
            else:
                # 为图像序列时，保存为视频
                self.total_frames = len(self.saved_images)
                height, width, layers = self.saved_images[0].shape
                size = (width, height)
                out = cv2.VideoWriter('./output_video_' + str(now_time) + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30,
                                      size)
                for img in self.saved_images:
                    out.write(img)
                    self.cur_frames += 1
                    # 更新进度条
                    if self.progressBar:
                        self.progressBar.setValue(self.cur_frames / self.total_frames * 100)
                out.release()

            # 保存识别结果为csv
            headers = [self.tableWidget.horizontalHeaderItem(i).text() for i in range(self.tableWidget.columnCount())]
            nrows = self.tableWidget.rowCount()
            ncols = self.tableWidget.columnCount()
            # Prepare to save velocity data
            velocity_data = {
                'Track ID': [],
                'Velocity X': [],
                'Velocity Y': [],
                'Class': [],
                'Elapsed Time': []
            }
            # 初始化数据字典，包括速度数据
            data = {header: [] for header in headers + ['Track ID']}
            for i in range(nrows):
                NO_id = self.tableWidget.item(i, 0).text()  # 假设跟踪ID存储在第一列
                velocities = self.velocity_history.get(NO_id, [(0, 0)])  # 默认为 (0, 0) 如果没有历史数据
                velocity_x, velocity_y = velocities[-1] if velocities else (0, 0)  # 获取最新速度，或默认为0
                for j in range(ncols):
                    item_text = self.tableWidget.item(i, j).text() if self.tableWidget.item(i, j) else ''
                    data[headers[j]].append(item_text)
                data['Track ID'].append(NO_id)
            for track_id, velocities in self.velocity_history.items():
                for vx, vy, cls, elapsed_time in velocities:
                    velocity_data['Track ID'].append(track_id)
                    velocity_data['Velocity X'].append(vx)
                    velocity_data['Velocity Y'].append(vy)
                    velocity_data['Class'].append(cls)
                    velocity_data['Elapsed Time'].append(elapsed_time)
            # 打印即将保存的数据来进行检查
            for track_id, velocities in self.velocity_history.items():
                for vx, vy, cls, elapsed_time in velocities:
                    print(f"Track ID {track_id}: Velocity X={vx}, Velocity Y={vy}, Class={cls}, Time={elapsed_time}")
            df = pd.DataFrame(data)
            # Create DataFrame from velocity data
            df_velocity = pd.DataFrame(velocity_data)

            # 添加一行包含ratio的数据
            # 创建一个DataFrame来存储ratio信息
            ratio_data = pd.DataFrame([{headers[0]: "Image Resize Ratio", headers[1]: self.ratio}])

            # 将字典转换为DataFrame并附加到原有的DataFrame上
            df = pd.concat([df, ratio_data], ignore_index=True)

            df.to_csv('./table_data_' + str(now_time) + '.csv', index=False)
            df_velocity.to_csv(f'./velocity_data_{now_time}.csv', index=False)

            QMessageBox.information(self.centralwidget, "Save file",
                                    "\nSuccessed!\nCurrent video is saved！\nexcel data is saved as csv dile！")
        else:
            QMessageBox.warning(self.centralwidget, "Save file", "saving...\nFailed!\nCheck your operation！")

    def update_tooltip(self, value):
        """更新滑块的toolTips为当前值."""
        sender = self.sender()
        if sender:
            sender.setToolTip(str((value + 1) / 100))
