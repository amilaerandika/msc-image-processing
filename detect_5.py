from ultralytics import YOLO
import cv2

# Load YOLOv8 pretrained model
model = YOLO('yolov8n-seg.pt')  # Or use yolov8s.pt for better accuracy

# Define COCO vehicle class IDs
vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# Load traffic video or use 0 for webcam
video_path = "video/traffic_video.avi"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict using YOLO
    results = model.predict(frame)

    annotated_frame = results[0].plot() 

    # Display result
    cv2.imshow("Vehicle Segmentation", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()