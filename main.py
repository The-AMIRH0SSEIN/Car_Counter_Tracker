from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

x1 , y1 , x2 , y2 = 770, 340, 1050, 540  #counting_box location

model = YOLO("yolov8m.pt")

video_path = "car_traffic.mp4"

cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda:[])
car_list = []
while cap.isOpened():
    _, frame = cap.read()
    results = model.track(frame, persist=True)
    boxes = results[0].boxes.xywh.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    clss = results[0].boxes.cls.cpu().numpy()
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,100,0),2)



    for box, track_id, cls in zip(boxes, track_ids, clss):
        x, y, w, h = box
        track = track_history[track_id]
        if (track_id not in car_list and x > x1 and x < x2 and y > y1 and y < y2 and cls == 2):
            car_list.append(track_id)
        track.append((float(x), float(y)))
        if len(track) > 30:
            track.pop(0)

        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=False, color=(100, 255, 0), thickness=2)

    cv2.putText(frame,f'cars = {len(car_list)}',(770,320),cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
    cv2.imshow("Car Tracking", frame)

    if (cv2.waitKey(1) & 0xFF == ord("q")):
        break

cap.release()
cv2.destroyAllWindows()