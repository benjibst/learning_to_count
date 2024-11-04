import cv2
import json
import random
import matplotlib.pyplot as plt

with open("labels/labelled.json","r") as f:
    data = json.load(f)

rand = random.choices(list(data.keys()),k=9)
fig,ax = plt.subplots(3,3,figsize=(20,20))
for i in range(3):
    for j in range(3):
        img = cv2.imread(f"images/{rand[i*3+j]}")
        for box in data[rand[i*3+j]]:
            x,y,w,h = box
            cv2.rectangle(img,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,255,0),2)
        ax[i,j].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        ax[i,j].axis("off")
plt.show()

    