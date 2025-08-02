from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.predict(source="https://www.youtube.com/watch?v=KFTV-wcTCpw", show=True)

for r in results:
    print(r.boxes.cls)  # class IDs