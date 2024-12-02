import cv2
import json
import random
import matplotlib.pyplot as plt

with open("labels/test.json","r") as f:
    data = json.load(f)

rand = random.choices(list(data.keys()),k=9)
fig,ax = plt.subplots(3,3,figsize=(20,20))
for i in range(3):
    for j in range(3):
        img = cv2.resize(cv2.imread(f"images/{rand[i*3+j]}"),(640,640))
        for box in data[rand[i*3+j]]["boxes"]:
            x1,y1,x2,y2 = box
            cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        ax[i,j].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        ax[i,j].axis("off")
plt.show()

    