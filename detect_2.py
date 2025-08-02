from ultralytics import YOLO
import cv2

# Load YOLOv8 model (downloaded or pretrained)
model = YOLO('yolov8n.pt')  # use 'yolov8s.pt' or your own model if needed

# Load your image
image_path = 'bus.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Run prediction
results = model.predict(source=image, save=False, conf=0.25)

# Visualize and display the result using OpenCV
annotated_frame = results[0].plot()  # returns image with boxes and labels

cv2.imshow("Detected Image", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()