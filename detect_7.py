import cv2
from ultralytics import YOLO

# Load image
img_path = "bus.jpg"
img = cv2.imread(img_path)

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")

# Predict keypoints
results = model.predict(source=img, show=True, conf=0.5)
for r in results:
    keypoints = r.keypoints.xy  # shape [num_people, num_keypoints, 2]
    count=1
    for person in keypoints:
        print(f"Person {count} key points:")
        count += 1
        for kp in person:
            x, y = kp
            print(f"Keypoint: ({x:.1f}, {y:.1f})")

# Visualize and display the result using OpenCV
annotated_frame = results[0].plot()  # returns image with boxes and labels

cv2.imshow("Detected Image", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()