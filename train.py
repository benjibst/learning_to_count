#!python
import numpy as np
import os
import json
import cv2
import random
import sys

# keras backend jax
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import keras

os.environ["QT_QPA_PLATFORM"] = "eglfs"
import matplotlib.pyplot as plt


class DataLoaderFactory:
    def __init__(self):
        self.image_paths, self.labels = self.load_labelled_img_paths()
        self.n_labelled = len(self.labels)
        self.unlabelled_image_paths = self.load_unlabelled_img_paths(self.image_paths)
        self.image_size = self.get_image_size()

    def get_image_size(self):
        return cv2.imread(self.image_paths[0]).shape[0:2]

    def load_unlabelled_img_paths(self, labelled_image_paths):
        unlabelled_image_paths = []
        for img in os.listdir("images"):
            if img not in labelled_image_paths:
                unlabelled_image_paths.append("images/" + img)
        return unlabelled_image_paths

    def load_labelled_img_paths(self):
        with open("labels/labelled.json", "r") as f:
            labels = json.load(f)
        image_paths = []
        labels_list = []
        labelled = np.random.permutation(list(labels.keys()))
        for k in labelled:
            image_paths.append("images/" + k)
            labels_list.append(labels[k])
        print(f"Loaded {len(labels_list)} labels")
        return image_paths, labels_list

    def get_dataloader(self, n_samples):
        if n_samples > self.n_labelled:
            n_samples = self.n_labelled
        if n_samples == 0:
            raise ValueError("No labelled data available")
        iterator_images = self.image_paths[0:n_samples]
        iterator_labels = self.labels[0:n_samples]
        self.image_paths = self.image_paths[n_samples:]
        self.labels = self.labels[n_samples:]
        self.n_labelled -= n_samples
        return PedestrianDataIterator(
            self.image_size, image_paths=iterator_images, labels=iterator_labels
        )


def generate_heatmap(image_size, boxes):
    heatmap = np.zeros((image_size[0] // 8, image_size[1] // 8))
    for box in boxes:
        x, y, _, _ = box
        x = int(x // 8)
        y = int(y // 8)
        heatmap[y, x] = 1
    return heatmap


class PedestrianDataIterator(keras.utils.Sequence):
    def __init__(
        self, image_size, batch_size=8, image_paths=None, labels=None, **kwargs
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_paths = image_paths
        self.labels = labels
        super().__init__(**kwargs)

    def __len__(self):
        return int(len(self.image_paths) / self.batch_size)

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_labels = self.labels[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        return self.__data_generation(batch_image_paths, batch_labels)

    def __data_generation(self, batch_image_paths, batch_labels):
        X = np.empty((self.batch_size, *self.image_size, 1))
        heatmap_size = (self.image_size[0] // 8, self.image_size[1] // 8)
        y = np.empty((self.batch_size, *heatmap_size), dtype=float)

        for i, file_path in enumerate(batch_image_paths):
            img = (
                keras.utils.img_to_array(
                    keras.utils.load_img(
                        file_path,
                        target_size=self.image_size,
                        keep_aspect_ratio=True,
                        color_mode="grayscale",
                    )
                )
                / 255.0
            )  # Normalize the image to [0, 1]
            X[i,] = img
            y[i,] = generate_heatmap(img.shape[0:2], batch_labels[i])

        return X, y


data_loader = DataLoaderFactory()
n = data_loader.n_labelled
n_train = int(n * 3 / 4)
n_val = int(n * 1 / 8)
n_test = int(n * 1 / 8)
train_data_loader = data_loader.get_dataloader(n_train)
val_data_loader = data_loader.get_dataloader(n_val)
test_data_loader = data_loader.get_dataloader(n_test)


if os.path.exists("model.keras"):
    print("Loading model from file")
    model = keras.models.load_model("model.keras")

else:
    print("Creating new model")
    model = keras.Sequential(
        [
            keras.layers.InputLayer(shape=(224, 224, 1)),
            keras.layers.Conv2D(
                16,
                (5, 5),
                activation="relu",
                padding="same",
            ),
            keras.layers.Conv2D(
                16,
                (5, 5),
                activation="relu",
                padding="same",
            ),
            keras.layers.AvgPool2D((2, 2)),
            keras.layers.Conv2D(
                16,
                (5, 5),
                activation="relu",
                padding="same",
            ),
            keras.layers.Conv2D(
                16,
                (5, 5),
                activation="relu",
                padding="same",
            ),
            keras.layers.AvgPool2D((2, 2)),
            keras.layers.Conv2D(
                16,
                (5, 5),
                activation="relu",
                padding="same",
            ),
            keras.layers.Conv2D(
                16,
                (5, 5),
                activation="relu",
                padding="same",
            ),
            keras.layers.AvgPool2D((2, 2)),
            keras.layers.Conv2D(1, (1, 1), activation="relu", padding="same"),
        ]
    )

if len(sys.argv) == 3:
    model.compile(optimizer="adam", loss="mse")
    if sys.argv[1] == "train":
        print(f"Training model for {int(sys.argv[2])} epochs")

        model.fit(
            train_data_loader, epochs=int(sys.argv[2]), validation_data=val_data_loader
        )
        print("Testing model")
        model.evaluate(test_data_loader)
        model.save("model.keras")
    if sys.argv[1] == "infer":
        show = int(sys.argv[2])
        images = random.choices(data_loader.unlabelled_image_paths, k=show)
        print(f"Choosing from {len(data_loader.unlabelled_image_paths)} images")
        fig, ax = plt.subplots(show, 2)
        for i in range(show):
            img = keras.utils.load_img(
                images[i],
                target_size=(224, 224),
                keep_aspect_ratio=True,
                color_mode="grayscale",
            )
            img = keras.utils.img_to_array(img) / 255.0
            batch = img.reshape(1, 224, 224, 1)
            heatmap = model.predict(batch)[0]
            ax[i, 0].imshow(img)
            ax[i, 1].imshow(heatmap)
        plt.show()
