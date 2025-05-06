import cv2
import torch
import argparse

# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true', help='Save output video')
args = parser.parse_args()

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Initialize webcam capture
cap = cv2.VideoCapture(0)  # 0 for default webcam
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Get webcam properties for VideoWriter
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20  # Default FPS for webcams

# Initialize VideoWriter only if saving is enabled
out = None
if args.save_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_detection.mp4', fourcc, fps, 
                        (frame_width, frame_height))

# Detection settings
detect_enabled = True
RETAIL_CLASSES = ['person', 'bottle', 'handbag', 'tie', 'suitcase', 'cup']
COLORS = {
    'person': (0, 255, 0),      # Green
    'bottle': (255, 0, 0),      # Blue
    'handbag': (0, 0, 255),     # Red
    'default': (255, 255, 0)    # Yellow for others
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Toggle detection with 'd' key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        detect_enabled = not detect_enabled
    elif key == ord('q'):
        break

    if detect_enabled:
        # Convert frame to RGB and run inference
        results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Process detections
        for _, det in results.pandas().xyxy[0].iterrows():
            if det['name'] in RETAIL_CLASSES and det['confidence'] > 0.5:
                x1, y1, x2, y2 = map(int, det[['xmin', 'ymin', 'xmax', 'ymax']])
                color = COLORS.get(det['name'], COLORS['default'])
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, 
                          f"{det['name']}: {det['confidence']:.2f}",
                          (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.6, color, 2)

    # Display status
    status = "ON" if detect_enabled else "OFF"
    cv2.putText(frame, f"Detection: {status}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Write frame to output if saving enabled
    if out is not None:
        out.write(frame)
    
    # Display
    cv2.imshow("Retail Object Detection", frame)

# Clean up
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

if args.save_video:
    print("Processing complete. Output saved to 'output_detection.mp4'")
else:
    print("Processing complete (no output saved)")