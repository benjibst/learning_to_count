import json
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

plt.ion()
plt.show()  

try:
    with open("labels/_labels.json","r") as f:
        try:
            labels = json.load(f)
            print(f"Loaded {len(labels.keys())} labels")
        except json.JSONDecodeError:
            labels = {}
except FileNotFoundError:
    labels = {}
files = os.listdir("images")
for f in labels.keys():
    print(f"Removing {f}")
    files.remove(f)
files = np.random.permutation(files)
index = 0
while index<len(files): 
    img = mpimg.imread("images/"+files[index])
    plt.imshow(img)
    label = input("People: ")
    if(label == "q"):
        with open("labels/_labels.json","w") as f:
            json.dump(labels,f)
        print(f"Saved {len(labels.keys())} labels")
        exit()
    if(label == "p"):
        index -= 1
        continue
    if(label == "n"):
        index += 1
        continue
    try:
        labels[files[index]] = int(label)
        index = index + 1
    except ValueError:
        continue
    