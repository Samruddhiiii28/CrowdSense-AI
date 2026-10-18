import cv2
import math
from collections import defaultdict,deque
from ultralytics import YOLO

MODEL_PATH="yolo11n.pt"
CAMERA_INDEX=0

CONFIDENCE=0.35
HISTORY_LENGTH=20

model=YOLO(MODEL_PATH)

cap=cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error:could not open camera")
    exit()

track_history=defaultdict(lambda:deque(maxlen=HISTORY_LENGTH))

def calculate_direction(previous_point,current_point):
    if previous_point is None:
        return "STATIONARY"

    previous_x,previous_y=previous_point
    current_x,current_y=current_point

    dx=current_x-previous_x
    dy=current_y-previous_y

    movement_treshold=5

    if abs(dx)<movement_treshold and abs(dy)<movement_treshold:
        return "STATIONARY"

    if abs(dx)>abs(dy):

        if dx>0:
            return "RIGHT"
        else:
            return "LEFT"
    else:
        if dy>0:
            return "DOWN"
        else:
            return "UP"

def calculate_distance(previous_point,current_point):
    if previous_point is None:
        return 0

    x1,y1=previous_point
    x2,y2=current_point

    distance=math.sqrt((x2-x1)**2+(y2-y1)**2)
    return distance

while True:
    ret,frame=cap.read()
    if not ret:
        print("Error:Could not read frame")
        break

    results=model.track(frame,persist=True,classes=[0],conf=CONFIDENCE,tracker="bytetrack.yaml",verbose=False)
    annotated_frame=results[0].plot()
    boxes=results[0].boxes
    ttl_ppl=len(boxes)

    if boxes.id is not None:
        track_ids=boxes.id.int().cpu().tolist()
        coordinates=boxes.xyxy.cpu().numpy()

        for track_id, box in zip(track_ids,coordinates):
            x1,y1,x2,y2=box

            center_x=int((x1+x2)/2)
            center_y=int((y1+y2)/2)

            current_point=(center_x,center_y)

            previous_point=None

            if len(track_history[track_id])>0:
                previous_point=track_history[track_id][-1]

            track_history[track_id].append(current_point)

            direction=calculate_direction(previous_point,current_point)
            distance=calculate_distance(previous_point,current_point)
            cv2.circle(annotated_frame,current_point,5,(0,255,255),-1)

            cv2.putText(annotated_frame,f"ID{track_id}",(center_x+10,center_y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),0)
            cv2.putText(annotated_frame,direction,(center_x+10,center_y+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

            points=list(track_history[track_id])

            for i in range(1,len(points)):
                cv2.line(annotated_frame,points[i-1],points[i],(255,0,0),2)

        cv2.putText(annotated_frame,f"people:{ttl_ppl}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("CrowdSense AI-Crowd Flow",annotated_frame)

        if cv2.waitKey(1) & 0xFF==ord("q"):
            break

cap.release()
cv2.destroyAllWindows()