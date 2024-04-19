from moviepy.editor import VideoFileClip, concatenate_videoclips

# 加载视频文件
clip1 = VideoFileClip("E:\Python_Projects/YOLO_Bytetrack/runs/JapanStreet.mp4")
clip2 = VideoFileClip("E:/Python_Projects/YOLO_Bytetrack/runs/Final_Prediction/Final_Prediction.mp4")

# 拼接视频
final_clip = concatenate_videoclips([clip1, clip2])

# 输出拼接后的视频文件
final_clip.write_videofile("E:\Python_Projects/YOLO_Bytetrack/runs/CombinedProject.mp4")
