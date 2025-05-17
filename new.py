import cv2
import torch
import urllib.request
import os
import numpy as np

scale_factor = None
calibrated = False
calibration_distance = 2.0  # meters (known real-world distance)

# Download MiDaS model if not already present
model_path = "weights/model-small.onnx"
if not os.path.exists(model_path):
    print("Downloading MiDaS model...")
    os.makedirs("weights", exist_ok=True)
    url = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# Load MiDaS model
midas_model = cv2.dnn.readNet(model_path)
midas_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
midas_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not accessible.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # ==== Run MiDaS Depth Estimation ====
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (256, 256),
                                 mean=[0.485, 0.456, 0.406],
                                 swapRB=True, crop=False)
    midas_model.setInput(blob)
    depth_map = midas_model.forward()[0]
    # Resize to match original frame
    depth_map = cv2.resize(depth_map, (width, height))

# --- POST-PROCESSING START ---
# 1. Apply Median Blur to smooth noisy spikes
    depth_map_blurred = cv2.medianBlur(depth_map.astype(np.float32), 5)

# 2. Normalize to 0–255 (8-bit) range
    depth_map_normalized = cv2.normalize(depth_map_blurred, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_normalized = depth_map_normalized.astype(np.uint8)

# 3. Apply Color Map for visualization
    depth_map_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)
# --- POST-PROCESSING END ---

    # ==== Run YOLOv5 ====
    results = model(frame)
    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        class_name = row['name']
        conf = row['confidence']  # Get confidence score

        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        z = int(depth_map[cy, cx])
        
        # Perform one-time calibration when key 'c' is pressed
        if calibrated and scale_factor:
          z_meters = round(scale_factor * z, 2)  # Convert to meters
        else:
          z_meters = "?"

        # ==== Draw 3D Cube (Cuboid) ====
        box_width = x2 - x1
        box_height = y2 - y1
        scale = 1000 / (z + 100)
        offset_x = int(box_width * 0.3 * scale)
        offset_y = int(box_height * 0.3 * scale)

        # Front face
        pt1 = (x1, y1)
        pt2 = (x2, y1)
        pt3 = (x2, y2)
        pt4 = (x1, y2)

        # Back face
        bpt1 = (x1 + offset_x, y1 + offset_y)
        bpt2 = (x2 + offset_x, y1 + offset_y)
        bpt3 = (x2 + offset_x, y2 + offset_y)
        bpt4 = (x1 + offset_x, y2 + offset_y)

        color = (0, int(z * 2) % 255, 255 - int(z) % 255)

        # Draw front and back faces
        cv2.line(frame, pt1, pt2, color, 2)
        cv2.line(frame, pt2, pt3, color, 2)
        cv2.line(frame, pt3, pt4, color, 2)
        cv2.line(frame, pt4, pt1, color, 2)
        cv2.line(frame, bpt1, bpt2, color, 2)
        cv2.line(frame, bpt2, bpt3, color, 2)
        cv2.line(frame, bpt3, bpt4, color, 2)
        cv2.line(frame, bpt4, bpt1, color, 2)

        # Connect corners
        cv2.line(frame, pt1, bpt1, color, 2)
        cv2.line(frame, pt2, bpt2, color, 2)
        cv2.line(frame, pt3, bpt3, color, 2)
        cv2.line(frame, pt4, bpt4, color, 2)

        label = f"{class_name} ({cx},{cy},{z}) - Conf: {conf:.2f}"  # Show confidence score
        # Update label to show distance in meters
        llabel = f"{class_name} ({cx},{cy}) - {z_meters}m - Conf: {conf:.2f}"
        cv2.putText(frame, llabel, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Combine original frame with depth map
    combined_display = np.hstack((frame, depth_map_color))
    if calibrated:
      cv2.putText(combined_display, "Calibration Active",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
      cv2.putText(combined_display, "Press 'c' to Calibrate",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("YOLO + MiDaS 3D Cubes + Depth(Post-processing)", combined_display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
      break

    elif key == ord('c'):
      if not detections.empty:
        row = detections.iloc[0]
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        z_pred = depth_map[cy, cx]

        if z_pred > 0:
            scale_factor = calibration_distance / z_pred
            calibrated = True
            print(f"[CALIBRATION SUCCESS] Scale factor = {scale_factor:.4f}")
        else:
            print("[CALIBRATION FAILED] Invalid depth value.")


cap.release()
cv2.destroyAllWindows()