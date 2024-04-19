import os
from PIL import Image
from Prediction import images_to_video
import re
def overlay_images_top_bottom_Reverse(upper_image_path, lower_image_path, output_path):
    """
    将两个输入的图像叠加在一起，第一个输入的图像会被扩展或裁剪以匹配第二个图像的大小，然后叠加在底层。

    参数:
    upper_image_path: 上层图像的文件路径。
    lower_image_path: 下层图像的文件路径。
    output_path: 输出图像的文件路径。
    """
    # 打开下层图像并确保它是 RGBA 模式，以便处理透明度
    lower_image = Image.open(lower_image_path).convert("RGBA")
    lower_width, lower_height = lower_image.size

    # 打开上层图像并确保它也是 RGBA 模式
    upper_image = Image.open(upper_image_path).convert("RGBA")
    upper_width, upper_height = upper_image.size

    # 如果上层图像的尺寸不同，则调整上层图像的尺寸以匹配下层图像
    if upper_width != lower_width or upper_height != lower_height:
        upper_image = upper_image.resize((lower_width, lower_height), Image.LANCZOS)

    # 将上层图像叠加在下层图像上
    combined_image = Image.alpha_composite(lower_image, upper_image)

    # 保存叠加后的图像
    combined_image.save(output_path)



def batch_overlay_images_Reverse(frames_dir, base_image_path, output_dir):
    """
    将指定目录中的所有图片作为上层，覆盖在同一个下层图片上，并保存结果。

    参数:
    frames_dir: str - 包含上层图像的目录。
    base_image_path: str - 作为下层的单个图像路径。
    output_dir: str - 输出目录，用于保存合成后的图像。
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有的图片文件
    frame_files = [f for f in sorted(os.listdir(frames_dir)) if f.endswith('.png')]
    # 按文件名中的数字排序
    frame_files.sort(key=lambda x: int(re.search(r'(\d+)', x).group()))

    # 对每一张图片执行叠加操作
    for frame_file in frame_files:
        overlay_image_path = os.path.join(frames_dir, frame_file)
        output_image_path = os.path.join(output_dir, frame_file)
        # 将目录中的图像作为上层，指定的图像作为下层
        overlay_images_top_bottom_Reverse(overlay_image_path, base_image_path, output_image_path)
        print(f"Overlayed image saved to {output_image_path}")

input_dir='E:/Python_Projects/YOLO_Bytetrack/runs/Masks/PredictionFrames/Clear/'
base_path='E:/Python_Projects/YOLO_Bytetrack/runs/fixing_frame.png'
output_dir='E:/Python_Projects/YOLO_Bytetrack/runs/Final_Prediction/'
batch_overlay_images_Reverse(input_dir,base_path,output_dir)
output_file='E:/Python_Projects/YOLO_Bytetrack/runs/Final_Prediction/Final_Prediction.mp4'
images_to_video(output_dir, output_file, fps=30)
