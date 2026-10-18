import cv2
from ultralytics import YOLO

def get_density_level(count):
    if count<=5:
        return "Low"
    elif count<=15:
        return "Medium"
    else:
        return "High"
    
model=YOLO("yolo11n.pt")

cap=cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret,frame=cap.read()

    if not ret:
        print("could not read frame")
        break

    results=model.track( frame,persist=True,classes=[0],conf=0.35,tracker="bytetrack.yaml",verbose=False)

    annotated_frame=results[0].plot()

    height,width=frame.shape[:2]

    zone_width=width//2
    zone_height=height//2

    zone_counts={"A":0,"B":0,"C":0,"D":0}
    boxes=results[0].boxes
    for box in boxes:
        x1,y1,x2,y2=box.xyxy[0].cpu().numpy()

        center_x=int((x1+ x2)/2)
        center_y=int((y1+y2)/2)

        if center_x < zone_width and center_y < zone_height:
            zone_counts["A"] += 1

        elif center_x >= zone_width and center_y < zone_height:
            zone_counts["B"] += 1

        elif center_x < zone_width and center_y >= zone_height:
            zone_counts["C"] += 1

        else:
            zone_counts["D"] += 1

    cv2.line(annotated_frame,(zone_width,0),(zone_width,height),(255,255,255),2)
    cv2.line(annotated_frame,(0,zone_height),(width,zone_height),(255,255,255),2)

    total_ppl=sum(zone_counts.values())

    zone_density={zone:get_density_level(count) for zone,count in zone_counts.items()}
    cv2.putText(annotated_frame,f"Total ppl:{total_ppl}",(20,height-20),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

    cv2.putText(annotated_frame,f"Zone A:{zone_counts['A']} ({zone_density['A']})",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(annotated_frame,f"Zone B:{zone_counts['B']} ({zone_density['B']})",(width//2+20,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.putText(annotated_frame,f"Zone C:{zone_counts['C']} ({zone_density['C']})",(20,zone_height+40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    cv2.putText(annotated_frame,f"Zone D:{zone_counts['D']} ({zone_density['D']})",(width//2+20,zone_height+40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    
    cv2.imshow(
        "CrowdSense AI-crowd Analytics",annotated_frame
    )

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()