from PIL import Image
from Prediction import remove_white_parts
import sys  # 导入sys模块，用于处理Python运行时环境的一些操作
import cv2  # 导入OpenCV库，用于处理图像和视频
from QtFusion.path import abs_path
from QtFusion.handlers import MediaHandler  # 从QtFusion库中导入MediaHandler类，用于处理媒体数据
from QtFusion.utils import get_cls_color  # 从QtFusion库中导入get_cls_color函数，用于获取类别颜色
from PySide6 import QtWidgets, QtCore  # 导入PySide6库的QtWidgets和QtCore模块，用于创建GUI和处理Qt的核心功能
from YOLOv8v5TrackModel import YOLOv8v5Tracker  # 从YOLOv8Model模块中导入YOLOv8Detector类，用于进行YOLOv8物体检测
from run_test_video import MainWindow  # 从QtFusion库中导入FBaseWindow类，用于创建主窗口
import time
from datasets.PersonCar.label_name import Label_list
from collections import defaultdict
import numpy as np
from QtFusion.utils import drawRectBox  # 从QtFusion库中导入drawRectBox函数，用于在图像上绘制矩形框
from Prediction import overlay_images_top_bottom
from Prediction import images_to_video
import os
import re

track_history = defaultdict(list)  # 初始化轨迹历史字典
retain_frames = 50
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


            if track_id is not None:
                center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))  # 计算边界框的中心点
                track_history[track_id].append(center)  # 更新轨迹历史
                if len(track_history[track_id]) > retain_frames:  # 保留最近retain_frames个轨迹点
                    track_history[track_id].pop(0)

                # 绘制轨迹线
                if len(track_history[track_id]) > 1:
                    track_color = colors[cls_id]  # 获取与标记框相同的颜色
                    cv2.polylines(image, [np.array(track_history[track_id], np.int32).reshape((-1, 1, 2))], False, track_color, 2)

            label = '%s %.0f%%' % (name, conf * 100)  # 创建标签，包含类别名称和置信度
            image = drawRectBox(image, bbox, alpha=0.2, addText=label, color=colors[cls_id])  # 在图像上绘制边界框和标签
            # 计数每个类别的个数
            class_counts[name] += 1
    if all(count == 0 for count in class_counts.values()):
        # 确保保存帧的目录存在
        if not os.path.exists(output_frames_dir):
            os.makedirs(output_frames_dir)

        # 构造输出图片的文件名，这里假设每次调用该函数的时候传入的image参数都是连续的
        frame_id = len(track_history)  # 或其他递增的ID生成方式
        frame_filename = f"frame_{frame_id:04d}.png"
        frame_path = os.path.join(output_frames_dir, frame_filename)

        # 保存帧
        cv2.imwrite(frame_path, image)
        print(f"Saved frame without detections to {frame_path}")

    # 构建并显示统计信息
    stats_text = "Counts: " + ", ".join([f"{k}: {v}" for k, v in class_counts.items()])
    window.stats_label.setText(stats_text)  # 更新统计信息标签的文本

    window.dispImage(window.label, image)  # 在窗口的label上显示图像
    # 将处理后的帧写入输出视频
    out_video.write(image)

def remove_black_parts(image):
    """将图片中的黑色或接近黑色的部分转换为透明"""
    image = image.convert("RGBA")  # 确保图片是 RGBA 模式，以便处理透明度
    datas = image.getdata()

    newData = []
    for item in datas:
        # 修改下面的阈值来调整什么被认为是“接近黑色”的
        if item[0] < 50 and item[1] < 50 and item[2] < 50:  # 检查黑色或接近黑色的像素
            newData.append((255, 255, 255, 0))  # 使其完全透明
        else:
            newData.append(item)

    image.putdata(newData)
    return image


def video_to_frames(video_path, output_dir):
    """
    将视频文件逐帧转换为图片。

    参数:
    video_path: str - 视频文件的路径。
    output_dir: str - 保存帧图片的目录。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    # 逐帧读取视频
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束或读取错误

        # 构造输出图片的文件名
        frame_filename = os.path.join(output_dir, f"frame_{frame_id:04d}.png")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved {frame_filename}")
        frame_id += 1

    # 释放视频捕获对象
    cap.release()
    print("Done converting video to frames.")
def make_region_transparent(image_path, region_coordinates):
    """
    将图像中指定的区域转换为透明，并调整为与指定图像相同的尺寸。

    参数:
    image_path: 指定的图像文件路径，用于获取目标尺寸
    region_coordinates: 指定区域的坐标，格式为 [x1, y1, x2, y2]，左上角和右下角坐标

    返回:
    处理后的 PIL Image 对象
    """
    # 加载指定路径的图像以获取目标尺寸
    target_image = Image.open(image_path)

    # 获取目标尺寸并打印
    target_width, target_height = target_image.size
    print("原始图像尺寸:", target_image.size)

    # 创建一个与目标图像相同尺寸的黑色图像
    transparent_image = Image.new("RGBA", (target_width, target_height), color=(0, 0, 0))

    # 获取指定区域的坐标
    x1, y1, x2, y2 = region_coordinates

    # 将指定区域内的像素设为透明
    for y in range(y1, y2):
        for x in range(x1, x2):
            if 0 <= x < target_width and 0 <= y < target_height:
                transparent_image.putpixel((x, y), (255, 255, 255))

    # 调整透明图像的尺寸为与指定图像相同，并打印调整后的尺寸
    edited_image = transparent_image.resize((target_width, target_height))
    print("调整后的透明图尺寸:", edited_image.size)

    return edited_image

# 新的函数
def batch_overlay_images(frames_dir, overlay_image_path, output_dir):
    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all image files in the frames directory
    frame_files = [f for f in sorted(os.listdir(frames_dir)) if f.endswith('.png')]

    # Sort the files by the number in the file name
    frame_files = sorted(frame_files, key=lambda x: int(re.search(r'(\d+)', x).group()))

    # Overlay each image and save the result
    for frame_file in frame_files:
        base_image_path = os.path.join(frames_dir, frame_file)
        output_image_path = os.path.join(output_dir, frame_file)
        # Call the overlay function with the upper image being the constant overlay
        # and the lower image being each image from the directory
        overlay_images_top_bottom(overlay_image_path, base_image_path, output_image_path)
        print(f"Overlayed image saved to {output_image_path}")

# 指定图像文件路径和指定区域的坐标
image_path = 'E:/Python_Projects/YOLO_Bytetrack/runs/last_frame.png'
region_coordinates = [1179, 780, 1299, 1020]  # 指定区域的左上角和右下角坐标

# 将指定区域转换为透明，并调整为与指定图像相同的尺寸
edited_image = make_region_transparent(image_path, region_coordinates)

# 保存处理后的图像
output_path = 'E:/Python_Projects/YOLO_Bytetrack/runs/mask.png'
edited_image.save(output_path)
result_image = remove_white_parts(edited_image)
outputF_path = 'E:/Python_Projects/YOLO_Bytetrack/runs/result_mask.png'
result_image.save(outputF_path)

# 输出处理后的图像尺寸
print("处理后的图像尺寸:", edited_image.size)
print("最终的图像尺寸:", result_image.size)

if __name__ == '__main__':  # 如果当前模块是主模块
    reverse_video_path = 'E:/Python_Projects/YOLO_Bytetrack/runs/reversed_JapanStreet.mp4'
    framefolder_path = 'E:/Python_Projects/YOLO_Bytetrack/runs/Frames'
    #可不用
    video_to_frames(reverse_video_path,framefolder_path)

    frames_dir='E:/Python_Projects/YOLO_Bytetrack/runs/Frames/'
    overlay_image_path='E:/Python_Projects/YOLO_Bytetrack/runs/result_mask.png'
    output_dir='E:/Python_Projects/YOLO_Bytetrack/runs/Frame_with_mask/'
    #可不用
    batch_overlay_images(frames_dir, overlay_image_path, output_dir)

    input_folder=output_dir
    output_file='E:/Python_Projects/YOLO_Bytetrack/runs/Frame_with_mask/Mask_video.mp4'
    images_to_video(input_folder, output_file, fps=30)
    output_frames_dir = 'E:/Python_Projects/YOLO_Bytetrack/runs/No_Detections_Frames/'
    cls_name = Label_list
    reverse_video_path=output_file
    cap = cv2.VideoCapture(reverse_video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    # 创建YOLOv8v5Tracker对象
    model = YOLOv8v5Tracker()  # 创建YOLOv8Detector对象
    model.load_model(abs_path("E:/Python_Projects/YOLO_Bytetrack/runs/detect/train_v5_PersonCar/weights/best.pt", path_type="current"))  # 加载自训练的YOLOv5模型
    colors = get_cls_color(model.names)  # 获取类别颜色
    out_video_path = abs_path("output_video.mp4", path_type="current")  # 定义输出视频的路径
    # 使用H.264编码器和MP4格式
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 也可以尝试使用 'avc1'
    out_video_path = "output_video.avi"  # 确保文件扩展名与格式匹配
    out_video = cv2.VideoWriter(out_video_path, fourcc, 30.0, (850, 500))
    app = QtWidgets.QApplication(sys.argv)  # 创建QApplication对象
    window = MainWindow()  # 创建MainWindow对象
    filename = abs_path(reverse_video_path, path_type="current")  # 定义视频文件的路径
    videoHandler = MediaHandler(fps=30)  # 创建MediaHandler对象，设置帧率为30fps
    videoHandler.frameReady.connect(frame_process)  # 当有新的帧准备好时，调用frame_process函数进行处理
    videoHandler.setDevice(filename)  # 设置视频源
    videoHandler.startMedia()  # 开始处理媒体

    # 显示窗口
    window.show()



    # 指定图片的路径
    Finalimage_path = 'E:/Python_Projects/YOLO_Bytetrack/runs/No_Detections_Frames/frame_0002.png'

    # 检查图片是否存在
    if os.path.exists(Finalimage_path):
        # 打开图片
        image = Image.open(Finalimage_path)
        # 应用remove_black_parts函数
        modified_image = remove_black_parts(image)
        # 可以选择保存修改后的图片或进行其他处理
        final_path='E:/Python_Projects/YOLO_Bytetrack/runs/No_Detections_Frames/Finalframe_0001.png'
        modified_image.save(final_path)  # 保存修改后的图片覆盖原文件
        print("Final Image processed and saved!!!")
        output_FINAL_path='E:/Python_Projects/YOLO_Bytetrack/runs/fixing_frame.png'
        overlay_images_top_bottom(final_path,image_path,output_FINAL_path)

    else:
        print("Failed to find the resource")

    # 进入 Qt 应用程序的主循环
    sys.exit(app.exec())
    out_video.release()