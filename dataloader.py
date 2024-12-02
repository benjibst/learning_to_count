import cv2,json,keras
import numpy as np

def generate_heatmap(orig_size,heatmap_size, labels):
    heatmap = np.zeros(heatmap_size)
    divx = orig_size[1] / heatmap_size[1]
    divy = orig_size[0] / heatmap_size[0]
    for i,cls in enumerate(labels["classes"]):
        if(cls == "person"):
            x1, y1, x2, y2 = labels["boxes"][i]
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            x = int(x // divx)
            y = int(y // divy)
            heatmap[y, x] = 1
    return heatmap


class PedestrianDataIterator(keras.utils.Sequence):
    def __init__(self, image_size,labels_file, batch_size=8, heatmap_div=4,trained_size=(640,640)):
        self.batch_size = batch_size
        self.image_size = image_size
        self.orig_size = trained_size
        self.load_labels(labels_file)
        self.heatmap_div = heatmap_div
        self.heatmap_size = (image_size[0] // self.heatmap_div, image_size[1] // self.heatmap_div)
        super().__init__()

    def load_labels(self,labels_file):
        with open(labels_file,"r") as f:
            self.labels_dict = json.load(f)
        self.image_paths = list(self.labels_dict.keys())
        self.labels = [self.labels_dict[i] for i in self.image_paths]

    def __len__(self):
        return int(len(self.image_paths) / self.batch_size)

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size : (index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size : (index + 1) * self.batch_size]
        return self.__data_generation(batch_image_paths, batch_labels)

    def __data_generation(self, batch_image_paths, batch_labels):
        X = np.empty((self.batch_size, *self.image_size, 1))
        y = np.empty((self.batch_size, *self.heatmap_size), dtype=float)

        for i, file_path in enumerate(batch_image_paths):
            img = keras.utils.img_to_array(
                keras.utils.load_img(
                    f"images/{file_path}",
                    color_mode="grayscale",
                    target_size=self.image_size,
                    keep_aspect_ratio=True,
                )
            )/255.0
            img = np.reshape(img, (*self.image_size, 1))


            X[i,] = img
            y[i,] = generate_heatmap(self.orig_size,self.heatmap_size, batch_labels[i])
        return X, y
    
    def __data_generation_unlabelled(self, img_path):
        X = np.empty((1,*self.image_size, 1))
        img = (
            keras.utils.img_to_array(
                keras.utils.load_img(
                    img_path,
                    target_size=self.image_size,
                    keep_aspect_ratio=True,
                    color_mode="grayscale",
                )
            )
        )
        X[0,] = img
        return X
    
    def representative_data_gen(self):
        for i in self.image_paths:
            yield [self.__data_generation_unlabelled(f"images/{i}").astype(np.float32)]
    
