import cv2
import json
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
if False:
    base = "/home/benni/dev/learning_to_count_data/"
else: 
    base = "/home/benjamin/learning_to_count_data/"
labels = {}
with open(f"{base}labels/train.json","r") as f:
    labels.update(json.load(f))
with open(f"{base}labels/val.json","r") as f:
    labels.update(json.load(f))
with open(f"{base}labels/test.json","r") as f:
    labels.update(json.load(f))
print(f"Loaded {len(labels)} labels")

n_rows = 2
n= n_rows**2
curr = 0
files = [x for x in list(labels.keys()) if x.startswith("kaggle")]
indices = np.random.permutation(len(files))
while True:
    try:
        images = []
        for i in range(n):
            curr+=1
            file = files[indices[curr]]
            label = labels[file]
            img = cv2.imread(f"{base}images/{file}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for k,cls in enumerate(label["classes"]):
                x1,y1,x2,y2 = label["boxes"][k]
                if cls == "person":
                    color = (0,255,0)
                elif cls == "car":
                    color = (255,0,0)
                else:
                    color = (0,0,255)
                cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
            images.append(img)
        fig = plt.figure(figsize=(15,15))
        grid = ImageGrid(fig, 111, 
                        nrows_ncols=(n_rows, n_rows),  # creates 2x2 grid of axes
                        axes_pad=0,  # pad between axes
                        )

        for ax, im in zip(grid, images):
            ax.imshow(im)
        plt.show()
        
    except Exception as e:
        print(e)
        plt.close()
        break


    