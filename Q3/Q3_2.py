import cv2
import torch
from PIL import Image

# Load YOLOv5 model (automatically downloads on first run)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # 'yolov5s' = small version
model.eval()  # Set to evaluation mode

# Initialize webcam (or video file)
cap = cv2.VideoCapture(0)  # 0 = default webcam, or pass video path

# Optional: Toggle detection with 'd' key
detect_enabled = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Toggle detection
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        detect_enabled = not detect_enabled

    if detect_enabled:
        # Convert frame to RGB (YOLOv5 expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = model(rgb_frame)
        
        # Parse results
        detections = results.pandas().xyxy[0]  # DataFrame with columns: xmin, ymin, xmax, ymax, confidence, class, name
        
        # Filter retail objects (COCO classes: person=0)
        RETAIL_CLASSES = ['person', 'bottle', 'chair', 'cup', 'laptop']  # Add more retail-related classes as needed
        for _, det in detections[detections['name'].isin(RETAIL_CLASSES)].iterrows():
            x1, y1, x2, y2 = map(int, det[['xmin', 'ymin', 'xmax', 'ymax']])
            conf = det['confidence']
            cls = det['name']
            
            # Draw bounding box and label
            color = (0, 255, 0)  # Green for all objects (customize per class)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display frame
    cv2.imshow("Retail Object Detection", frame)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Add before the while loop
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_detection.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Inside the loop, after drawing boxes
out.write(frame)

# After the loop, release VideoWriter
out.release()