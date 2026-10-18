import cv2

video_path="data/raw/crowd.mp4"

cap=cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error:Could not open video")
    exit()

width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=cap.get(cv2.CAP_PROP_FPS)
total_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration=total_frame/fps if fps>0 else 0

print("========== VIDEO INFORMATION ==========")
print("Width:", width)
print("Height:", height)
print("FPS:", fps)
print("Total Frames:", total_frame)
print("Duration:", round(duration, 2), "seconds")
print("=======================================")

cap.release()