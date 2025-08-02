import cv2
from ultralytics import YOLO

# Load image
img_path = "https://ultralytics.com/images/boats.jpg"


# Load YOLOv8 pose model
model = YOLO("yolov8n-obb.pt")

# Predict keypoints
results = model.predict(source=img_path, show=True)

# Visualize and display the result using OpenCV
annotated_frame = results[0].plot()  # returns image with boxes and labels

cv2.imshow("Detected Image", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()