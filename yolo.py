import torch
import cv2
import numpy as np
from ultralytics import YOLO
# Load the YOLOv5 model (by default it loads the YOLOv5s small model, you can choose others)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'

# Open the webcam
cap = cv2.VideoCapture(0)  # Change the number if your camera isn't at index 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the current frame
    results = model(frame)  # Runs inference on the frame

    detections = results.pandas().xyxy[0]  # x1, y1, x2, y2, confidence, class, name

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']

        # Calculate center coordinates
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw center point
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        # Put label and coordinates
        text = f"{label} ({cx},{cy}) {confidence:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


    # Display the frame
    cv2.imshow("YOLO Object Detection", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
