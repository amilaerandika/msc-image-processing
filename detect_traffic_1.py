from ultralytics import YOLO
import cv2

# Load YOLOv8 pretrained model
model = YOLO('yolov8n.pt')  # Or use yolov8s.pt for better accuracy

# Define COCO vehicle class IDs
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Load traffic video or use 0 for webcam
video_path = "video/traffic_video.avi"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict using YOLO
    results = model.predict(frame, verbose=False)

    # Annotate only vehicle detections
    annotated = frame.copy()
    vehicle_count = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id in vehicle_classes:
                vehicle_count += 1
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls_id]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Print number of vehicles in the current frame
    print(f"Number of vehicles in this frame: {vehicle_count}")

    # Display result
    cv2.imshow("Vehicle Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()