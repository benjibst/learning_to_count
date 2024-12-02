import cv2
import json
import random
import matplotlib.pyplot as plt

labels = {}
with open("labels/train.json","r") as f:
    labels.update(json.load(f))
with open("labels/val.json","r") as f:
    labels.update(json.load(f))
with open("labels/test.json","r") as f:
    labels.update(json.load(f))
print(f"Loaded {len(labels)} labels")

n_rows = 2
n= n_rows**2
while True:
    try:
        rand = random.choices(list(labels.keys()),k=n)
        fig,ax = plt.subplots(n_rows,n_rows,figsize=(20,20))
        for i in range(n_rows):
            for j in range(n_rows):
                img = cv2.resize(cv2.imread(f"images/{rand[i*n_rows+j]}"),(640,640))
                curr_label = labels[rand[i*n_rows+j]]
                for k,cls in enumerate(curr_label["classes"]):
                    x1,y1,x2,y2 = curr_label["boxes"][k]
                    if cls == "person":
                        color = (0,255,0)
                    elif cls == "car":
                        color = (0,0,255)
                    else:
                        color = (255,0,0)
                    cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
                ax[i,j].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
                ax[i,j].axis("off")
        plt.show()
        
    except Exception as e:
        print(e)
        plt.close()
        break


    