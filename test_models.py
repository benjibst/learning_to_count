import tensorflow_hub as hub
import keras
import keras_cv
import tensorflow as tf
import os
import cv2
import ultralytics
import numpy as np
import matplotlib.pyplot as plt
# Apply image detector on a single image.
os.environ['TFHUB_CACHE_DIR'] = '/home/benni/tf_cache'
def run_tf_models(model, x):
    LABEL_MAP = {
        0: "unlabeled",
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic",
        11: "fire",
        12: "street",
        13: "stop",
        14: "parking",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        26: "hat",
        27: "backpack",
        28: "umbrella",
        29: "shoe",
        30: "eye",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports",
        38: "kite",
        39: "baseball",
        40: "baseball",
        41: "skateboard",
        42: "surfboard",
        43: "tennis",
        44: "bottle",
        45: "plate",
        46: "wine",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted",
        65: "bed",
        66: "mirror",
        67: "dining",
        68: "window",
        69: "desk",
        70: "toilet",
        71: "door",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        83: "blender",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy",
        89: "hair",
        90: "toothbrush",
        91: "hair",
        92: "banner",
        93: "blanket",
        94: "branch",
        95: "bridge",
        96: "building",
        97: "bush",
        98: "cabinet",
        99: "cage",
        100: "cardboard",
        101: "carpet",
        102: "ceiling",
        103: "ceiling",
        104: "cloth",
        105: "clothes",
        106: "clouds",
        107: "counter",
        108: "cupboard",
        109: "curtain",
        110: "desk",
        111: "dirt",
        112: "door",
        113: "fence",
        114: "floor",
        115: "floor",
        116: "floor",
        117: "floor",
        118: "floor",
        119: "flower",
        120: "fog",
        121: "food",
        122: "fruit",
        123: "furniture",
        124: "grass",
        125: "gravel",
        126: "ground",
        127: "hill",
        128: "house",
        129: "leaves",
        130: "light",
        131: "mat",
        132: "metal",
        133: "mirror",
        134: "moss",
        135: "mountain",
        136: "mud",
        137: "napkin",
        138: "net",
        139: "paper",
        140: "pavement",
        141: "pillow",
        142: "plant",
        143: "plastic",
        144: "platform",
        145: "playingfield",
        146: "railing",
        147: "railroad",
        148: "river",
        149: "road",
        150: "rock",
        151: "roof",
        152: "rug",
        153: "salad",
        154: "sand",
        155: "sea",
        156: "shelf",
        157: "sky",
        158: "skyscraper",
        159: "snow",
        160: "solid",
        161: "stairs",
        162: "stone",
        163: "straw",
        164: "structural",
        165: "table",
        166: "tent",
        167: "textile",
        168: "towel",
        169: "tree",
        170: "vegetable",
        171: "wall",
        172: "wall",
        173: "wall",
        174: "wall",
        175: "wall",
        176: "wall",
        177: "wall",
        178: "water",
        179: "waterdrops",
        180: "window",
        181: "window",
        182: "wood"
    }
    
    img_sz = x.shape[:2]

    out = model(x.reshape((1,*x.shape)))
    # filter all detections with score > 0.6 and return the boxes and classes of those detections
    boxes = out["detection_boxes"].numpy()
    scores = out["detection_scores"].numpy()
    classes = out["detection_classes"].numpy()
    high_scores = scores>0.5
    boxes = boxes[high_scores]
    classes = [LABEL_MAP[int(x)] for x in classes[high_scores]]
    scores = scores[high_scores]
    boxes[:,[0,1]] = boxes[:,[1,0]]
    boxes[:,[2,3]] = boxes[:,[3,2]]
    boxes[:,0] *= img_sz[0]
    boxes[:,1] *= img_sz[1]
    boxes[:,2] *= img_sz[0]
    boxes[:,3] *= img_sz[1]
    return {"boxes":boxes,"scores":scores,"classes":classes}

class EfficientDet:
    def __init__(self):
        self.model = hub.load("https://www.kaggle.com/models/tensorflow/efficientdet/TensorFlow2/d0/1")
    def run(self,x):
        return run_tf_models(self.model,x)
    
class SSDMobileNetV2:
    def __init__(self):
        self.model = hub.load("https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/TensorFlow2/ssd-mobilenet-v2/1")
    def run(self,x):
        return run_tf_models(self.model,x)

class CenterNetHourglass:
    def __init__(self):
        self.model = hub.load("https://www.kaggle.com/models/tensorflow/centernet-hourglass/TensorFlow2/1024x1024/1")
    def run(self,x):
        x_ = np.zeros((1,1024,1024,3))
        x_big = cv2.resize(x,(1024,1024))
        x_[0] = x_big
        return run_tf_models(self.model,x)
    
class FasterRCNNInceptionResnetv2:
    def __init__(self):
        self.model = hub.load("https://www.kaggle.com/models/tensorflow/faster-rcnn-inception-resnet-v2/TensorFlow2/640x640/1")
    def run(self,x):
        return run_tf_models(self.model,x)

def run_yolomodel(model,x):
    out = model.predict(x)
    boxes = out[0].boxes.xyxy
    scores = out[0].boxes.conf
    names = model.names
    classes = [names[int(x)] for x in out[0].boxes.cls]
    return {"boxes":boxes,"scores":scores,"classes":classes}

class Yolo:
    def __init__(self,yolo_path):
        self.model = ultralytics.YOLO(yolo_path)
    def run(self,x):
        return run_yolomodel(self.model,x)
        
base = "/home/benni/dev/learning_to_count_data"
images_base = f"{base}/images"
image_files = [f"{images_base}/{x}" for x in os.listdir(images_base)]
rand_idx = np.random.permutation(len(image_files))
curr = 0

def get_random_img_path(n):
    global curr
    if curr + n >= len(image_files):
        curr = 0
    ret = [image_files[i] for i in rand_idx[curr:curr+n]]
    curr += n
    return ret

models = [Yolo("yolo11x.pt"),FasterRCNNInceptionResnetv2(),EfficientDet(),SSDMobileNetV2()]
rows = 3
img_sz = (500,500)
while True:
    try:
        img_paths = get_random_img_path(rows)
        imgs = [keras.utils.load_img(x,target_size=img_sz) for x in img_paths]
        print()
        for i,img in enumerate(imgs):
            plt.subplot(len(models)+1,rows,i+1)
            plt.imshow(img)
        imgs_in = [keras.utils.img_to_array(x) for x in imgs]
        for m,model in enumerate(models):
            for i in range(len(imgs)):
                out = model.run(imgs_in[i])
                print(out)
                #copy image and draw boxes
                img_cp = np.copy(imgs_in[i]).astype(np.uint8)
                for box,c,s in zip(out["boxes"],out["classes"],out["scores"]):
                    x1,y1,x2,y2 = box
                    if c == "person":
                        cv2.rectangle(img_cp,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
                    elif c == "car":
                        cv2.rectangle(img_cp,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                    else:
                        cv2.rectangle(img_cp,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),2)
                plt.subplot(len(models)+1,rows,(m+1)*rows+i+1)
                plt.imshow(img_cp)
        plt.show()
    except KeyboardInterrupt:
        break
