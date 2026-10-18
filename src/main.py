import cv2
import os
import time

from tracking.person_tracker import PersonTracker
from analytics.crowd_density import CrowdDensity

video_path="data/raw/crowd.mp4"
output_path="outputs/videos/main_pipeline.mp4"

os.makedirs("outputs/videos",exist_ok=True)

tracker=PersonTracker()
density=CrowdDensity()

cap=cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Could not open video")
    exit()

width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps=cap.get(cv2.CAP_PROP_FPS)

fourcc=cv2.VideoWriter_fourcc(*"mp4v")
out=cv2.VideoWriter(output_path,fourcc,video_fps,(width,height))

print("total frames in video:",cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("fps:",video_fps)

while True:
    start_time=time.time()
    ret,frame=cap.read()

    if not ret:
        break

    person_count,annotated_frame,results=tracker.track(frame)

    zone_counts,zone_density=density.calculate_zones(results.boxes, width,height)
    annotated_frame=density.draw_zones(annotated_frame,zone_counts,zone_density,width,height)

    end_time=time.time()
    processing_time=end_time-start_time
    fps=1/processing_time if processing_time>0 else 0

    cv2.putText(annotated_frame, f"FPS:{fps:.2f}",(20,height-60),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    out.write(annotated_frame)

    display_frame=cv2.resize(annotated_frame,(960,540))
    cv2.imshow("CrowdSense AI-main Pipeline",display_frame)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


