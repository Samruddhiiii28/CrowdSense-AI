import  cv2
from ultralytics import YOLO

MODEL_PATH="yolo11n.pt"
CAMERA_INDEX=0

CONFIDENCE=0.35

LINE_POSITION_RATIO=0.50

model=YOLO(MODEL_PATH)

cap=cv2.VideoCapture(CAMERA_INDEX )
if not cap.isOpened():
    print("Error:Could not open camera")
    exit()

previous_positions={}
counted_ids=set()

entry_count=0
exit_count=0

while True:
    ret,frame=cap.read()

    if not ret:
        print("Error:Could not read Frame")
        break

    height,width=frame.shape[:2]

    line_y=int(height*LINE_POSITION_RATIO)

    results=model.track(frame,persist=True,classes=[0],conf=CONFIDENCE,tracker="bytetrack.yaml",verbose=False)

    annotated_frame=results[0].plot()
    boxes=results[0].boxes

    if boxes.id is not None:
        track_ids=boxes.id.int().cpu().tolist()
        coordinates=boxes.xyxy.cpu().numpy()

        for track_id,box in zip(track_ids,coordinates):
            x1,y1,x2,y2=box

            center_x=int((x1+x2)/2)
            center_y=int((y1+y2)/2)

            current_position=center_y
            previous_position=previous_positions.get(track_id)

            if previous_position is not None:
                if(previous_position < line_y and current_position>=line_y):
                    if track_id not in counted_ids:
                        entry_count +=1

                        counted_ids.add(track_id)
                    elif(previous_position > line_y and current_position<=line_y):
                        if track_id not in counted_ids:
                            exit_count+=1

                        counted_ids.add(track_id)

        previous_positions[track_id]=current_position
        cv2.circle(annotated_frame,(center_x,center_y),5,(0,255,255),-1)

        cv2.putText(annotated_frame,f"ID{track_id}",(center_x+10,center_y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),2)
        cv2.line(annotated_frame,(0,line_y),(width,line_y),(255,255,255),3)
        cv2.putText(annotated_frame,"COUNTING LINE",(20,line_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

        currented_ppl=len(boxes)
        net_change=entry_count-exit_count

        cv2.putText(annotated_frame,f"Current_ppl:{currented_ppl}",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(annotated_frame,f"Entered:{entry_count}",(20,75),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(annotated_frame,f"Exited:{exit_count}",(20,110),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        cv2.putText(annotated_frame,f"Net Change:{net_change}",(20,145),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)


        cv2.imshow("CrowdSense AI-Entry exit",annotated_frame)
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break


cap.release()
cv2.destroyAllWindows()