import urllib.request
import os
import numpy as np
import time
import cv2
import keras
import sys

out_dir = "images_test"
in_dir = "dataraw"

def get_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith(".tif") or file.endswith(".jpg"):
                files.append(os.path.join(r, file))
    return files


if False:
    os.system(
        "wget http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz -O compressed.tar.gz"
    )
    os.mkdir("dataraw")
    os.system("tar -xf compressed.tar.gz -C dataraw")
    os.remove("compressed.tar.gz")
    os.mkdir("images")
files = get_files(in_dir)

for i in range(len(files)):
    print(f"{i+1}/{len(files)}")
    filenamejpg = str(i) + ".jpg"
    try:
        img = keras.utils.load_img(
            files[i],
            target_size=(224, 224),
            keep_aspect_ratio=True,
            color_mode="grayscale",
        )
        img.save(f"{out_dir}/{filenamejpg}")
    except:
        print("Error with file", files[i])
        continue
print("Done!")
