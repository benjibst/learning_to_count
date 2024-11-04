from ultralytics import YOLO
import cv2
import os
import json

base = "/home/benni/dev/learning_to_count"
images_base = f"{base}/images"
image_files = {f"{images_base}/{x}" for x in os.listdir(images_base)}
tot_files = len(image_files)

model = YOLO("yolo11x.pt", task="predict")
model.to("cuda")

with open(f"{base}/labels/labelled.json", "r") as f:
    labelled = json.load(f)
    print(f"Loaded {len(labelled)} labelled images")

for labelled_img in labelled.keys():
    image_files.remove(f"{images_base}/{labelled_img}")

try:
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        result = model(img, imgsz=640)[0]
        person_detections = []
        for box in result.boxes:
            obj_type = int(box.cls[0])
            if obj_type == 0:
                x, y, w, h = box.xywh[0]
                person_detections.append([int(x), int(y), int(w), int(h)])
        labelled[img_path.split("/")[-1]] = person_detections
except KeyboardInterrupt:
    print(f"Interrupted at {i}/{tot_files} images")
finally:
    with open("labels/labelled.json", "w") as f:
        json.dump(labelled, f, indent=4)
    print(f"Saved {len(labelled)}/{tot_files} ({len(labelled)*100/tot_files}%) images")
