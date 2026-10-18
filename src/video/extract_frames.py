import cv2
import os

video_path="data/raw/crowd.mp4"
output_dir="data/processed/frames"

os.makedirs(output_dir, exist_ok=True)

cap=cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("cant open video, check path")
    exit()
frame_no=0
saved_frame=0

while True:
    ret,frame=cap.read()

    if not ret:
        break

    if frame_no %30==0:
        filename=f"{output_dir}/frame_{saved_frame:04d}.jpg"
        cv2.imwrite(filename,frame)
        saved_frame+=1
    frame_no+=1
cap.release()
print("total frames in video:", frame_no)
print("frames saved:", saved_frame)
