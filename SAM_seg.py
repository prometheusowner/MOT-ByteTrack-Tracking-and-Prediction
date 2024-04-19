import tkinter as tk
from tkinter import filedialog, Canvas, Scrollbar, Frame
from PIL import Image, ImageTk ,ImageDraw
# 创建主窗口实例
root = tk.Tk()
root.geometry("900x830")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

# Initialfize your model
sam_checkpoint = "E:\SAM_checkpoint\sam_vit_l_0b3195.pth"
model_type = "vit_l"
cuda_available = torch.cuda.is_available()
torch_version = torch.__version__

# 打印PyTorch版本和CUDA可用性
print(f"PyTorch version: {torch_version}")
print(f"CUDA available: {cuda_available}")
cuda_version = torch.version.cuda
print(f"Current CUDA version: {cuda_version}")
# 根据CUDA可用性选择设备
device = "cuda" if cuda_available else "cpu"
print(f"Using device: {device}")

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,  # Requires open-cv to run post-processing
)

def create_scrollable_canvas(root):
    canvas = Canvas(root)
    scrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    frame = Frame(canvas)  # This frame will contain your image
    canvas_frame = canvas.create_window((0, 0), window=frame, anchor="nw")

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", on_frame_configure)

    def on_canvas_configure(event):
        canvas.itemconfig(canvas_frame, width=event.width)

    canvas.bind("<Configure>", on_canvas_configure)


    # 获取画布初始尺寸
    canvas_width = root.winfo_width()
    canvas_height = root.winfo_height()

    return canvas, scrollbar, frame, canvas_width, canvas_height

# 在初始化画布时获取尺寸
canvas, scrollbar, image_frame, canvas_width, canvas_height = create_scrollable_canvas(root)



# Place the canvas and scrollbar in the window
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Your image processing and model prediction functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_canvas_points(coords, labels, canvas, scale_x, scale_y, marker_size=5):
    for i in range(len(coords)):
        x, y = coords[i]
        label = labels[i]
        #print('Input_real_coord')
        #print(x)
        #print(y)

        #print('Show_point_scales')
        #print(scale_x)
        #print(scale_y)

        # 应用缩放比例
        x_scaled = x/scale_x
        y_scaled = y/scale_y
        #print('Show_point_real_coord')
        #print(x_scaled)
        #print(y_scaled)

        # 根据标签选择颜色
        color = 'green' if label == 1 else 'red'

        # 计算标记点的边界坐标，考虑缩放后的坐标
        x0 = x_scaled - marker_size
        y0 = y_scaled - marker_size
        x1 = x_scaled + marker_size
        y1 = y_scaled + marker_size

        # 在Canvas上绘制标记点
        canvas.create_oval(x0, y0, x1, y1, fill=color, outline='white')


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))






# Your UI setup, including buttons for navigation


current_mask_index = 0
current_mask_result_index = 0
masks = []
scores = []
image = None
masks = []
scores = []
input_point = None
input_label = None
canvas_image_ref = None
canvas_mask_ref = None
text_element = None
original_image_size = None  # 原始图像尺寸
displayed_image_size = None  # 显示在画布上的图像尺寸
#input_point = np.array([[270, 240]])  # 示例，可根据需要修改
#input_label = np.array([1])   # 示例，可根据需要修改
label = tk.Label(image_frame)
label.pack(side="bottom", fill="both", expand="yes")

def resize_image_to_fit_canvas(image, canvas_width, canvas_height):
    img_width, img_height = image.size
    ratio = min((canvas_width) / img_width, (canvas_height) / img_height)
    new_size = (int(img_width * ratio), int(img_height * ratio))
    return image.resize(new_size, Image.Resampling.LANCZOS)

def upload_and_process_image():
    global image, masks, scores, input_point, input_label, canvas, canvas_image_ref, canvas_mask_ref , original_image_size, displayed_image_size

    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_size = (image.shape[1], image.shape[0])  # 宽度, 高度
        print("Original image size:", original_image_size)
        img_pil = Image.fromarray(image)

        # 调整图片大小以适应画布
        root.update()  # 更新界面以确保画布尺寸已渲染
        canvas_width = canvas.winfo_width() or 800
        canvas_height = canvas.winfo_height() or 600
        if canvas_width > 0 and canvas_height > 0:
            img_pil = resize_image_to_fit_canvas(img_pil, canvas_width, canvas_height)
        displayed_image_size = img_pil.size  # 获取调整后的图像尺寸
        print("Displayed image size:", displayed_image_size)  # 调试打印
        # 更新Canvas上的图片
        img_tk = ImageTk.PhotoImage(image=img_pil)
        if canvas_image_ref is not None:
            canvas.delete(canvas_image_ref)
        canvas_image_ref = canvas.create_image(0, 0, anchor='nw', image=img_tk)
        canvas.image = img_tk

        # 重置相关变量
        masks = []
        scores = []
        input_point = None
        input_label = None
        current_mask_index = 0

        # 设置Canvas点击事件
        canvas.bind("<Button-1>", on_canvas_click)

def on_canvas_click(event):
    global input_point, input_label, image, original_image_size, displayed_image_size, scale_x, scale_y
    print("Canvas clicked")
    print("Original Click Coordinates:", event.x, event.y)
    if displayed_image_size is None :
        print("displayed_image_size NONE")
        pass
        if original_image_size is None:
            print("original_image_size NONE")
            pass



    # 计算坐标缩放比例
    scale_x = (original_image_size[0] / displayed_image_size[0])
    scale_y = (original_image_size[1] / displayed_image_size[1])
    #print('click_scale_inputpoint')
    #print(scale_x)
    #print(scale_y)

    # 调整点击坐标
    x, y = event.x * scale_x, event.y * scale_y
    print("Adjusted Click Coordinates: {:.1f}, {:.1f}".format(x, y))
    input_point = np.array([[x, y]])
    input_label = np.array([1])  # 假设这里我们总是将点击的点视为正类

    update_image_with_points()

def update_image_with_points():
    global image, input_point, input_label, canvas, canvas_image_ref, scale_x, scale_y

    if image is not None and input_point is not None:
        # 转换图像为PIL格式，假设image是NumPy数组
        img_pil = Image.fromarray(image)

        # 获取画布尺寸
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # 调整图像大小以适应画布
        img_pil = resize_image_to_fit_canvas(img_pil, canvas_width, canvas_height)

        # 将PIL图像转换为Tkinter可用的ImageTk.PhotoImage对象
        img_tk = ImageTk.PhotoImage(image=img_pil)

        # 在Canvas上显示图像
        if canvas_image_ref is not None:
            canvas.delete(canvas_image_ref)
        canvas_image_ref = canvas.create_image(0, 0, anchor='nw', image=img_tk)
        canvas.image = img_tk  # 保持对img_tk的引用

        # 显示标记点（如果有的话）
        if input_point is not None:
            # 计算缩放比例

            show_canvas_points(input_point, input_label, canvas,scale_x,scale_y)

def confirm_and_predict():
    global image, input_point, input_label, masks, scores
    if image is not None and input_point is not None:
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        show_mask_image(0, image)

def show_mask_image(index, image):
    global canvas, canvas_image_ref, img_tk
    current_mask_index = 0
    mask = masks[index]
    score = scores[index]
    # 定义天蓝色的RGB值
    sky_blue_color = np.array([135, 206, 237], dtype=np.uint8)

    # 创建一个RGB颜色的遮罩图像
    mask_rgb_color = np.stack([mask for _ in range(3)], axis=-1) * sky_blue_color

    # 将遮罩颜色应用到原始图像上，这里我们使用了0.5的alpha值来使遮罩半透明
    combined_image = np.where(mask_rgb_color, ((0.7 * mask_rgb_color) + (0.5 * image)).astype(np.uint8), image)

    # 将组合图像转换为PIL图像
    img_pil = Image.fromarray(combined_image)

    # 获取画布尺寸
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    # 调整图像大小以适应画布
    img_pil = resize_image_to_fit_canvas(img_pil, canvas_width, canvas_height)

    # 将PIL图像转换为Tkinter可用的ImageTk.PhotoImage对象
    imgtk = ImageTk.PhotoImage(image=img_pil)

    # 删除旧的画布图像，并在相同位置创建新图像
    if canvas_image_ref is not None:
        canvas.delete(canvas_image_ref)
    canvas_image_ref = canvas.create_image(0, 0, anchor='nw', image=imgtk)
    canvas.image = imgtk  # 保持对imgtk的引用




    # 显示标记点
    show_canvas_points(input_point, input_label, canvas,scale_x,scale_y)

    # 在绘制新文本之前删除旧的文本元素
    if text_element is not None:
        canvas.delete(text_element)

    update_textbox_text(f"Mask {index + 1}, Score: {score:.3f}")
    # 添加遮罩编号和分数文本

def update_textbox_text(new_text):
    text_var.set(new_text)

def next_mask():
    global current_mask_index, masks, image  # 声明image为全局变量
    if current_mask_index < len(masks) - 1:
        current_mask_index += 1
        show_mask_image(current_mask_index, image)


def previous_mask():
    global current_mask_index, image  # 声明image为全局变量
    if current_mask_index > 0:
        current_mask_index -= 1
        show_mask_image(current_mask_index, image)

def reset_program():
    global image, masks, scores, input_point, input_label, canvas_image_ref, canvas_mask_ref, original_image_size, displayed_image_size

    # 重置全局变量
    image = None
    masks = []
    scores = []
    input_point = None
    input_label = None
    current_mask_index=0
    original_image_size = None
    displayed_image_size = None
    canvas_image_ref = None
    canvas_mask_ref = None

    # 清除画布和标签内容
    canvas.delete("all")
    label.config(image="")
    label.image = None

def display_image_and_annotations():
    global masks,image, canvas_image_ref, canvas, scale_x, scale_y, current_mask_result_index
    if image is not None:
        masks = mask_generator.generate(image)
    else:
        print("No Image upload!")
        return
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    print(type(masks))
    f=Figure(figsize = (20, 20))
    plt.figure(figsize=(20, 20))
    plt.imshow(image)  # 假设 'image' 是你要显示的图像
    show_anns(masks)  # 假设 'masks' 是注释列表
    plt.axis('off')
    plt.savefig('D:\study of ECE 2022\ECE 442\mask_image\Seg_All_Big\_Result.jpg')
    plt.savefig('D:\study of ECE 2022\ECE 442\mask_image\Seg_All_Big\Previous_result\_Result-{}.jpg'.format(current_mask_result_index+1))
    current_mask_result_index +=1
    plt.show()
    image_path = 'D:\study of ECE 2022\ECE 442\mask_image\Seg_All_Big\_Result.jpg'
    output_path = 'D:\study of ECE 2022\ECE 442\mask_image\Seg_All_Big\_Result.jpg'
    remove_white_background(image_path, output_path, tolerance=200)
    # 读取图片
    img_pil = Image.open('D:\study of ECE 2022\ECE 442\mask_image\Seg_All_Big\_Result.jpg')

    # 获取Canvas的宽度和高度
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    # 调整图片大小以适应Canvas
    img_pil_resized = resize_image_to_fit_canvas(img_pil, canvas_width, canvas_height)
    # 将PIL图像转换为Tkinter可用的ImageTk.PhotoImage对象
    img_tk = ImageTk.PhotoImage(image=img_pil_resized)

    # 在Canvas上显示图像
    if canvas_image_ref is not None:
        canvas.delete(canvas_image_ref)
    canvas_image_ref = canvas.create_image(0, 0, anchor='nw', image=img_tk)
    canvas.image = img_tk  # 保持对img_tk的引用

    save_all_segImage()

def remove_white_background(image_path, output_path, tolerance=200, save_format='PNG'):
    """
    移除图片中的白色背景。

    :param image_path: 输入图片的路径。
    :param output_path: 输出图片的路径。
    :param tolerance: 用于判断白色背景的容忍度（0-255）。数值越低，判定为白色背景的条件越严格。
    :param save_format: 输出图片的格式，默认为'PNG'。
    """
    img = Image.open(image_path).convert("RGBA")
    datas = img.getdata()

    new_data = []
    for item in datas:
        # 判断像素颜色是否为白色背景，这里假定RGB三个通道的值都高于tolerance为白色背景
        if item[0] > tolerance and item[1] > tolerance and item[2] > tolerance:
            # 将白色背景转换为透明（设置alpha通道为0）
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    img.putdata(new_data)
    img.save(output_path, save_format)

def save_all_segImage():
    delete_File_contents()
    for i, mask in enumerate(masks):
        mask = ~mask['segmentation']
        mask = mask + 255
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mask = mask.astype(np.uint8)
        res = cv2.bitwise_and(image, mask)
        res[res == 0] = 255
        plt.imshow(res)
        plt.savefig('E:/Python_Projects/YOLO_Bytetrack/runs/SAM_seg/_Number-{}.png'.format(i + 1))
        plt.show()

def delete_File_contents():
    import os
    import shutil

    # 设置您想要清空的文件夹路径
    folder_path = 'E:/Python_Projects/YOLO_Bytetrack/runs/SAM_seg'

    # 确认文件夹存在
    if os.path.exists(folder_path):
        # os.listdir(folder_path) 会列出所有文件和目录
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                # 如果是文件夹则使用shutil.rmtree，如果是文件则使用os.remove
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        print("The folder does not exist")


def clear_canvas_and_display_solid_color(canvas):
    global canvas_image_ref

    # 创建一个纯色的 PIL 图像
    solid_color_image = Image.new("RGB", (canvas.winfo_width(), canvas.winfo_height()), (255, 255, 255))

    # 将 PIL 图像转换为 Tkinter 可用的格式
    img_tk = ImageTk.PhotoImage(solid_color_image)

    # 如果之前有图像，则先删除
    if canvas_image_ref is not None:
        canvas.delete(canvas_image_ref)

    # 在 Canvas 上显示纯色背景图像
    canvas_image_ref = canvas.create_image(0, 0, anchor='nw', image=img_tk)
    canvas.image = img_tk  # 保持对 img_tk 的引用


# 创建一个Entry控件用于显示文本
text_var = tk.StringVar()  # 创建一个字符串变量
text_entry = tk.Entry(root, textvariable=text_var, state='readonly', readonlybackground='white', fg='black')
text_entry.pack(side="top", fill="x")

plt.close('all')
upload_button = tk.Button(root, text="Upload and Process Image", command=upload_and_process_image)
upload_button.pack()

confirm_button = tk.Button(root, text="Confirm and Predict", command=confirm_and_predict)
confirm_button.pack()

# 在主界面添加重置按钮
reset_button = tk.Button(root, text="Reset", command=reset_program)
reset_button.pack()



# 添加显示遮罩的按钮
Seg_all_button = tk.Button(root, text="Seg all elements", command=display_image_and_annotations)
Seg_all_button.pack()



prev_button = tk.Button(root, text="<< Previous", command=previous_mask)
prev_button.pack(side=tk.LEFT)

next_button = tk.Button(root, text="Next >>", command=next_mask)
next_button.pack(side=tk.RIGHT)

if __name__ == '__main__':
    print('Function Start!!!')

root.mainloop()


