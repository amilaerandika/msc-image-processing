from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.predict(source="video/traffic_video.avi", show=True, save=True)

for r in results:
    print(r.boxes.cls)  # class IDs
