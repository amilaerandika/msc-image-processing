from ultralytics import YOLO

# Load classification model
model = YOLO("yolov8n-cls.pt")  # or yolov8s-cls.pt

# Predict class of the image
results = model("bus.jpg")  # Replace with your image path

# Print prediction
print(results[0].probs.top1)      # index of predicted class
print(results[0].probs.top1conf)  # confidence
print(results[0].names)           # class label mapping