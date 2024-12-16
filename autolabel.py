import os
import json
import keras
from autolabel_models import FasterRCNNInceptionResnetv2

if True:
    base = "/home/benni/dev/learning_to_count_data"
else:
    base = "/home/benjamin/learning_to_count_data"

images_base = f"{base}/images"
labels_base = f"{base}/labels"
image_files_to_label = {f"{x}" for x in os.listdir(images_base)}
tot_files = len(image_files_to_label)

def remove_labelled(files_to_label):
    train,val,test = {},{},{}
    with open(f"{labels_base}/test.json", "r") as f:
        test = json.load(f)
    with open(f"{labels_base}/train.json", "r") as f:
        train = json.load(f)
    with open(f"{labels_base}/val.json", "r") as f:
        val = json.load(f)
    for d in (train,val,test):
        files = [x for x in list(d.keys()) if x.startswith("kaggle")]
        for f in files:
            d.pop(f,None)
    for d in (train,val,test):
        files = list(d.keys())
        for f in files:
            if f in files_to_label: #its already labelled
                files_to_label.remove(f)
            else: #its not in the images directory
                print(f"Warning: {f} not in images directory")
                d.pop(f,None)
    
    return train,val,test

def serialize_model_output(detections):
    return {"boxes":detections["boxes"].tolist(),"classes":detections["classes"]}

def save_new_labelled(new_labelled,train,val,test,split = (0.7,0.2,0.1)):
    labelled_files = list(new_labelled.keys())
    n = len(labelled_files)
    train_n = int(n*split[0])
    val_n = int(n*split[1])
    train_files = labelled_files[:train_n]
    val_files = labelled_files[train_n:train_n+val_n]
    test_files = labelled_files[train_n+val_n:]
    for i in train_files:
        train[i] = serialize_model_output(new_labelled[i])
    for i in val_files:
        val[i] = serialize_model_output(new_labelled[i])
    for i in test_files:
        test[i] = serialize_model_output(new_labelled[i])
    with open(f"{labels_base}/test.json", "w") as f:
        json.dump(test,f)
    with open(f"{labels_base}/train.json", "w") as f:
        json.dump(train,f)
    with open(f"{labels_base}/val.json", "w") as f:
        json.dump(val,f)
    

train,val,test = remove_labelled(image_files_to_label)
model = FasterRCNNInceptionResnetv2()  
print(f"Labelling images: {len(image_files_to_label)}/{tot_files}")
new_labelled = {}

try:
    for i, img_path in enumerate(image_files_to_label):
        img = keras.utils.img_to_array(keras.utils.load_img(f"{images_base}/{img_path}"))
        detections = model.run(img)
        new_labelled[img_path] = detections
        print(f"Labelled {i+1}/{len(image_files_to_label)} images")
except KeyboardInterrupt:
    print(f"Interrupted at {i}/{len(image_files_to_label)} images")
finally:
    save_new_labelled(new_labelled,train,val,test)
    print(f"Saved {len(new_labelled)}/{len(image_files_to_label)} ({len(new_labelled)*100/len(image_files_to_label):.2f}%) images")
