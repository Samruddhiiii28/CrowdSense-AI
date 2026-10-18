import cv2
from ultralytics import YOLO

model=YOLO("yolo11n.pt")

cap=cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error:could not open webcam")
    exit()

while True:
    ret,frame=cap.read()

    if not ret:
        print("Could not read Frame")
        break

#YOLO + BYTETRACK
    results=model.track(frame,persist=True,classes=[0],conf=0.35,tracker="bytetrack.yaml",verbose=False)
    annotated_frame=results[0].plot()

    person_count=len(results[0].boxes)
    cv2.putText(annotated_frame,f"People:{person_count}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow(
        "crowdsense Ai-Real Time Tracking",annotated_frame
    )

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()