import pandas as pd
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import cv2
import sys
import matplotlib.pyplot as plt
from PIL import Image
import os
cuda_available = torch.cuda.is_available()
torch_version = torch.__version__

# 打印PyTorch版本和CUDA可用性
print(f"PyTorch version: {torch_version}")
print(f"CUDA available: {cuda_available}")
cuda_version = torch.version.cuda
print(f"Current CUDA version: {cuda_version}")

def get_last_frame(csv_file_path):
    global Box
    df = pd.read_csv(csv_file_path)

    # 从后向前搜索“Result”列中包含'people'的元素
    people_rows = df[df['Result'].str.contains('people', case=False, na=False)]

    if not people_rows.empty:
        # 找到最后一次出现'people'的行
        last_people_row = people_rows.iloc[-1]

        # 假设“Position”列包含四个坐标值，由逗号分隔
        position = last_people_row['Position'].split(',')

        # 将这些坐标值转换为整数并输出为四维数组
        four_dimensional_array = [int(coord.strip()) for coord in position]
        print(four_dimensional_array)
    else:
        print("No 'people' found in 'Result' column.")
    # 查找包含'Image Resize Ratio'的行
    ratio_row = df[df['N.O.'].str.contains('Image Resize Ratio', na=False)]

    if not ratio_row.empty:
        # 假设“Image Resize Ratio”在第一列，我们需要的值在第二列
        ratio_value = ratio_row.iloc[0, 1]  # 获取第二列的值
        print(ratio_value)
    else:
        print("没有找到'Image Resize Ratio'的相关信息。")
    ratio_value = float(ratio_value)
    Box = np.array(four_dimensional_array) / ratio_value
    print(Box)
    # 打开视频文件
    video_path = 'E:/Python_Projects/YOLO_Bytetrack/test_media/JapanStreet.mp4'
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否打开成功
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame = None

    # 跳转到视频的最后一帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)

    # 读取最后一帧
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        print("Error: Could not read frame.")
    else:
        # 显示最后一帧（测试用）
        # cv2.imshow('Last Frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存最后一帧为图片
        cv2.imwrite('E:/Python_Projects/YOLO_Bytetrack/runs/Last_Frame/last_frame.png', frame)
        cv2.imwrite('E:/Python_Projects/YOLO_Bytetrack/runs/Masks/last_frame.png', frame)

    # 释放视频对象
    cap.release()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def box_prompt():
    # 设置设备为CUDA（如果可用）
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 方框prompt SAM模型可以用一个方框作为输入，格式为[x1,y1,x2,y2],左上，右下。来进行单个目标的分割
    sys.path.append("..")
    sam_checkpoint = "E:\SAM_checkpoint\sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    # 加载图像
    image_path = 'E:/Python_Projects/YOLO_Bytetrack/runs/Last_Frame/last_frame.png'
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    input_box = np.array(Box)

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],  # 使用方框
        multimask_output=False,
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    show_mask(masks[0], ax)
    show_box(input_box, ax)
    plt.axis('off')

    # 保存分割图像到文件
    mask_output_path = 'E:/Python_Projects/YOLO_Bytetrack/runs/Masks/'
    plt.savefig(f"{mask_output_path}segmented_output.png")
    plt.show()
    segmentation_mask = masks[0]
    binary_mask = np.where(segmentation_mask > 0.5, 1, 0)
    white_background = np.ones_like(image) * 255
    new_image = white_background * (1 - binary_mask[..., np.newaxis]) + image * binary_mask[..., np.newaxis]
    new_image_pil = Image.fromarray(new_image.astype(np.uint8))
    new_image_pil.save('E:/Python_Projects/YOLO_Bytetrack/runs/Masks/clear_segmented_output.png')
    plt.imshow(cv2.cvtColor(new_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def remove_white_parts(image):
    """将图片中的白色或接近白色的部分转换为透明"""
    image = image.convert("RGBA")  # 确保图片是 RGBA 模式，以便处理透明度
    datas = image.getdata()

    newData = []
    for item in datas:
        # 修改下面的阈值来调整什么被认为是“接近白色”的
        if item[0] > 200 and item[1] > 200 and item[2] > 200:  # 检查白色或接近白色的像素
            newData.append((255, 255, 255, 0))  # 使其完全透明
        else:
            newData.append(item)

    image.putdata(newData)
    return image


def shift_image_area(img, dx, dy):
    """
    平移图像的像素。

    参数:
    img: 原始图像
    dx: 水平方向上的平移像素数
    dy: 垂直方向上的平移像素数

    返回:
    新的平移后的图像
    """
    # 将 PIL 图像对象 转换为 NumPy 数组
    img = np.array(img)
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])  # 平移矩阵
    translated_img = cv2.warpAffine(img, M, (cols, rows), borderValue=10)
    translated_img = Image.fromarray(translated_img)
    return translated_img


def overlay_images_top_bottom(upper_image_path, lower_image_path, output_path):
    """
    将两个输入的图像叠加在一起，第一个输入的图像会被扩展到与第二个图像相同的大小，然后叠加在底层。

    参数:
    upper_image_path: 第一个图像的文件路径
    lower_image_path: 第二个图像的文件路径
    output_path: 输出图像的文件路径
    """
    # 打开第一个图像并确保它是 RGBA 模式，以便处理透明度
    upper_image = Image.open(upper_image_path).convert("RGBA")

    # 打开第二个图像并确保它是 RGBA 模式，以便处理透明度
    lower_image = Image.open(lower_image_path).convert("RGBA")

    # 获取两个图像的尺寸
    upper_width, upper_height = upper_image.size
    lower_width, lower_height = lower_image.size

    # 如果上层图像比下层图像小，则将上层图像扩展到下层图像的大小
    if upper_width < lower_width or upper_height < lower_height:
        upper_image = upper_image.resize((lower_width, lower_height), Image.LANCZOS)

    # 将两个图像叠加在一起
    combined_image = Image.alpha_composite(lower_image, upper_image)

    # 保存叠加后的图像
    combined_image.save(output_path)

def images_to_video(input_folder, output_file, fps=30):
    # 获取文件夹中的所有图片文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # 检查文件夹中是否有图片文件
    if not image_files:
        print("No image files found in the input folder.")
        return

    # 对图片文件名按数字部分进行排序
    image_files_sorted = sorted(image_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

    # 读取第一张图片以确定视频帧的尺寸
    first_image = cv2.imread(os.path.join(input_folder, image_files_sorted[0]))
    frame_height, frame_width, _ = first_image.shape

    # 创建视频写入对象
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # 读取图片并写入输出视频
    for image_file in image_files_sorted:  # 按顺序读取图片文件
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)

    # 释放输出视频对象
    out.release()
    print(f"Output video saved as {output_file}")


def load_velocity_data_from_directory(directory):
    max_timestamp = -np.inf
    track_id_with_max_timestamp = None

    # 遍历目录中的所有 CSV 文件
    for filename in os.listdir(directory):
        if filename.startswith('velocity_data_') and filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            print(f"正在读取文件：{file_path}")
            try:
                data = pd.read_csv(file_path)
                # 打印数据头部来检查列名
                print(data.head())
                # 确保列名正常
                expected_columns = {'Track ID', 'Velocity X', 'Velocity Y', 'Class', 'Elapsed Time'}
                if not expected_columns.issubset(set(data.columns)):
                    print(f"列名不匹配。期望的列名: {expected_columns}")
                    continue

                # 查找所有被识别为人的记录，并找出时间戳最大的记录
                people_rows = data[data['Class'].str.lower() == 'people']
                if not people_rows.empty:
                    latest_people_row = people_rows.loc[people_rows['Elapsed Time'].idxmax()]
                    if latest_people_row['Elapsed Time'] > max_timestamp:
                        max_timestamp = latest_people_row['Elapsed Time']
                        track_id_with_max_timestamp = latest_people_row['Track ID']

            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
                continue

    # 如果找到有最大时间戳的'people'记录
    if max_timestamp > -np.inf:
        # 再次遍历数据，收集具有最大时间戳的'people'的速度数据
        velocity_array = []
        for filename in os.listdir(directory):
            if filename.startswith('velocity_data_') and filename.endswith('.csv'):
                file_path = os.path.join(directory, filename)
                try:
                    data = pd.read_csv(file_path)
                    if expected_columns.issubset(set(data.columns)):
                        # 筛选出特定 Track ID 且分类为 'people' 的速度数据
                        track_data = data[data['Track ID'] == track_id_with_max_timestamp]
                        people_track_data = track_data[track_data['Class'].str.lower() == 'people']
                        velocities = people_track_data[['Velocity X', 'Velocity Y']].values
                        for velocity in velocities:
                            velocity_array.append(velocity)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    continue

        return np.array(velocity_array)
    else:
        print("未找到任何符合条件的数据...")
        return np.array([])


def velocity_prediction(data):
    # 如果数据为空数组，则返回空数组
    if len(data) == 0:
        return np.array([])
    if len(data) >= 5:
        data = data[-5:]
    # 分别提取 x 和 y 方向的速度数据
    data_x = data[:, 0]
    data_y = data[:, 1]

    # 计算线性回归的斜率和截距
    x = np.arange(len(data_x))
    m, c = np.polyfit(x, data_x, 1)  # 线性拟合

    print("Slope (m):", m)
    print("Intercept (c):", c)

    # 绘制线性回归的结果
    plt.figure(figsize=(10, 6))
    plt.scatter(x, data_x, label='Velocity X')
    plt.plot(x, data_x * m + c, color='red', label='Linear Regression')

    plt.xlabel('Index')
    plt.ylabel('Velocity X')
    plt.title('Linear Regression of Velocity X')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 如果只有一个数据点，返回斜率作为预测
    if len(data_x) == 1:
        return m, 0  # 返回斜率和一个默认的截距值

    # 使用斜率进行预测
    next_index = len(data_x)  # 下一个索引为当前数据点数量
    next_prediction = np.array([next_index * m + c])

    return next_prediction



if __name__ == "__main__":
    for filename in os.listdir('E:/Python_Projects/YOLO_Bytetrack/'):
        if filename.startswith('table_data_') and filename.endswith('.csv'):
            csv_file_path = os.path.join('E:/Python_Projects/YOLO_Bytetrack/', filename)
    get_last_frame(csv_file_path)
    box_prompt()
    # 加载图像
    image_path = 'E:/Python_Projects/YOLO_Bytetrack/runs/Masks/clear_segmented_output.png'
    image = Image.open(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}. Please check the file path and integrity.")
        sys.exit(1)  # 退出程序
    # 去除背景
    result_image = remove_white_parts(image)
    result_image.save('E:/Python_Projects/YOLO_Bytetrack/runs/Masks/clear_segmented_output.png')
    # 获取整数坐标
    x1, y1, x2, y2 = np.array(Box).astype(int)
    directory_path = 'E:/Python_Projects/YOLO_Bytetrack/'
    velocity_array = load_velocity_data_from_directory(directory_path)
    print(velocity_array)

    # 预测速度
    prediction = velocity_prediction(velocity_array)

    # 根据返回结果的数量进行不同处理
    if len(prediction) == 2:
        dx, dy = prediction
    elif len(prediction) == 1:
        dx = prediction[0]
        dy = 0  # 设置 dy 为默认值
    else:
        print("Invalid prediction result.")
    # 如果 dy 的值小于 0.8，则将其设置为 0
    if dy < 0.8:
        dy = 0
    print(dx, dy)



    # 指定保存路径和迭代次数
    output_dir = 'E:/Python_Projects/YOLO_Bytetrack/runs/Masks/PredictionFrames/'
    num_iterations = 50

    #background = Image.open('E:/Python_Projects/YOLO_Bytetrack/runs/Last_Frame/last_frame.png')
    #overlay = Image.open('E:/Python_Projects/YOLO_Bytetrack/runs/Masks/clear_segmented_output.png')
    upper_path="E:/Python_Projects/YOLO_Bytetrack/runs/Masks/clear_segmented_output.png"
    base_path="E:/Python_Projects/YOLO_Bytetrack/runs/Last_Frame/last_frame.png"

    #result_image.show()
    print("Starting Processing Image")
    # 循环移动和保存图像
    for i in range(num_iterations):
        print(f"Processing shifted_image_{i}.png")
        save_path = f"{output_dir}clear/shifted_image_{i}.png"
        out_path = f"{output_dir}combined/shifted_image_{i}.png"
        result_image.save(save_path)  # 使用PIL的save方法保存图像
        overlay_images_top_bottom(save_path,base_path,out_path)
        result_image = shift_image_area(result_image, dx , dy )
        # 可选：展示每步结果
        #result_image.show()
        print(f"shifted_image_{i}.png is DONE")
    cv2.destroyAllWindows()

    # 指定输入文件夹和输出文件名
    input_folder1 = 'E:/Python_Projects/YOLO_Bytetrack/runs/Masks/PredictionFrames/Clear/'
    input_folder2 = 'E:/Python_Projects/YOLO_Bytetrack/runs/Masks/PredictionFrames/Combined/'
    clear_output_file = 'E:/Python_Projects/YOLO_Bytetrack/runs/Masks/PredictionFrames/Clear/output_video.mp4'
    combine_output_file = 'E:/Python_Projects/YOLO_Bytetrack/runs/Masks/PredictionFrames/Combined/output_video.mp4'
    # 调用函数将图片转换为视频
    images_to_video(input_folder1, clear_output_file)
    images_to_video(input_folder2, combine_output_file)

    print('System is DONE')










