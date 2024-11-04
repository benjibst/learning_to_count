import urllib.request
import os
import numpy as np
import time

test_split = 0.2
val_split = 0.2

def get_files(path):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file.endswith(".tif"):
                files.append(os.path.join(r, file))
    return files

os.system("wget http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz -O compressed.tar.gz")
os.mkdir("dataraw")
os.system("tar -xf compressed.tar.gz -C dataraw")
os.remove("compressed.tar.gz")
os.mkdir("images")
files = get_files("dataraw")

for i in range(len(files)):
    print(f"{i+1}/{len(files)}")
    filenamejpg = str(i) + ".jpg"
    os.system(f"ffmpeg -i {files[i]} images/{filenamejpg}")
os.system("rm -r dataraw")
print("Done!")
