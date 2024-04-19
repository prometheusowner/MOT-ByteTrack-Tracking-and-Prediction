import cv2

def reverse_video(input_video, output_video):
    # 打开视频文件
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return

    # 获取视频的总帧数和帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 获取视频的宽度和高度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入对象
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # 逐帧读取视频并写入输出视频
    for frame_num in range(total_frames - 1, -1, -1):  # 从最后一帧开始倒序
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

        # 显示当前处理进度
        print(f"Processing frame {total_frames - frame_num} / {total_frames}", end='\r')

    # 释放视频对象
    cap.release()
    out.release()

    print(f"Reversed video saved as {output_video}")

# 使用示例
input_video = 'E:/Python_Projects/YOLO_Bytetrack/runs/JapanStreet.mp4'
output_video = 'E:/Python_Projects/YOLO_Bytetrack/runs/reversed_JapanStreet.mp4'
reverse_video(input_video, output_video)
