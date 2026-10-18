import cv2

video_path="data/raw/crowd.mp4"

cap=cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error:Could not open video")
    exit()

while True:
    ret,frame=cap.read()
    if not ret:
        break
    cv2.imshow("Crowdsense-AI video",frame)

    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
cap.release() #closes video resources
cv2.destroyAllWindows()  #closes opencv display window
  