from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-obb.pt")  # load an official model


# Predict with the model sdfsdfsdf
results = model(source="https://www.youtube.com/watch?v=32gGo-IbsVU",show=True) 