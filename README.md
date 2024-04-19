# Enhancing Multi-Object Tracking with ByteTrack, YOLO, and SAM: Integrating Detection and Segmentation for Improved Accuracy and Efficiency

Versions:
Python -- 3.10;
PyTorch -- 2.22+CU118

HOW TO USE OUR CODE!!!
1.use run_main.py, choose your model to detect and tracking object. Click save putton to save the datalist and
2.use prediction.py, it will track the last people moving in the video, and predict his future motion
3.use reverse_video.py, it will reverse the video you pick
4.use reverse_detect.py, it will detect the area where the last person was, and get the RGB values when nothing is detected by yolo
5.use Final_production.py, it will produce the complete video of future prediction video
6.use video_clip.py, it will combine the original video and the predicted video, so a full combined video will be produced




REMEMBER to delete the old datalist CSV s to do a new prediction

run_test_camera.py or click the camera button in run_main.py 's UI, it will turn the camera on and do detect and track
run_test_video.py or click the video button in run_main.py 's UI, it will open the video and do detect and track
run_test_image.py or click the iamge button in run_main.py 's UI, it will open the image and do detect and track
run_train_model.py, Pick a pretrained model and use your own dataset to train it


SAM_seg.py, it contains a simple segment tool which can choose with all segment or point segment
you need to download checkpoints to use this function ( https://github.com/facebookresearch/segment-anything)

Device_test.py, it will test your environment to see if your GPU,CUDA is available

FOLDERS
datasets: our test train valid sets are here
icons: images files for UI
runs: our trained weight, processed images are all here
test_media: the media resource we use, E:\Python_Projects\YOLO_Bytetrack\test_media\JapanStreet.mp4 is mostly used
ultralytics: SAM, YOLO, bytetrack are here  https://github.com/ultralytics/ultralytics

Test Media files (videos) can be download from: https://drive.google.com/drive/folders/14IHEJlktm9PdCEe8ilP2UqdDtpCD7QU6?usp=drive_link

Download the zip file of the code: https://drive.google.com/file/d/1jn6idaUm_Hgvi4Fhomk9HZVw0oslunvT/view?usp=drive_link
