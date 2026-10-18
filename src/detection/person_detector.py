import cv2
import time
from ultralytics import YOLO

model=YOLO("yolo11n.pt")

video_path="data/raw/crowd.mp4"
output_path="outputs/videos/yolo_detection.mp4"
cap=cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video")
    exit()

width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps=cap.get(cv2.CAP_PROP_FPS)

fourcc=cv2.VideoWriter_fourcc(*"mp4v")

out=cv2.VideoWriter(output_path,fourcc,video_fps,(width,height))

print("total frames in video:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("fps:", cap.get(cv2.CAP_PROP_FPS))

frame_number = 0

while True:
    start_time=time.time()
    ret,frame=cap.read()
    if not ret:
        break

    results = model(frame, classes=[0], conf=0.35, verbose=False)

    person_count=len(results[0].boxes)
    annoated_frame=results[0].plot()

    end_time=time.time()
    processing_time=end_time-start_time

    fps=1/processing_time if processing_time >0 else 0
    display_frame = cv2.resize(annoated_frame, (960, 540))
    #people count
    cv2.putText( display_frame,f"People Count:{person_count}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    #FPS
   
    cv2.putText(annoated_frame,f"FPS:{fps:.2f}",(20,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    out.write(annoated_frame)
    cv2.imshow(
        "crowdsense AI-YOLO11 person Detection",
        display_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Detection Video saved to:")
print(output_path)