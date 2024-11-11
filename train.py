import numpy as np
import os
import random
import sys

import tensorflow as tf
# keras backend jax
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import keras
import matplotlib.pyplot as plt
from dataloader import DataLoaderFactory, UnlabelledDataIterator

img_dir = "images"
test_dir = "images_test"


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


def plot_model_input_output(model, data_loader,n,labelled=True):
    if n % 2 != 0:
        n += 1
    images_heatmaps = []
    choose = random.sample(range(len(data_loader)), n)
    for i in choose:
        if labelled:
            img, heatmap = data_loader.__getitem__(i)
        else:
            img = data_loader.__getitem__(i)
        heatmap = model.predict(img)[0]
        images_heatmaps.append(img[0])
        images_heatmaps.append(heatmap)
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = n // 2
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(images_heatmaps[i - 1])
    plt.show()

if len(sys.argv) == 3:
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    if sys.argv[1] in ("train", "infer"):
        data_loader = DataLoaderFactory()
        n = data_loader.n_labelled
        n_train = int(n * 3 / 4)
        n_val = int(n * 1 / 8)
        n_test = int(n * 1 / 8)
        train_data_loader = data_loader.get_dataloader(n_train)
        val_data_loader = data_loader.get_dataloader(n_val)
        test_data_loader = data_loader.get_dataloader(n_test)

        if sys.argv[1] == "train":
            print(f"Training model for {int(sys.argv[2])} epochs")
            model.fit(train_data_loader, epochs=int(sys.argv[2]), validation_data=val_data_loader)
            print("Testing model")
            model.evaluate(test_data_loader)
            model.save("model.keras")
        else:
            test_data_loader.batch_size = 1
            plot_model_input_output(model, test_data_loader, int(sys.argv[2]))

    if sys.argv[1] == "test":
        dataloader = UnlabelledDataIterator("images_test")
        plot_model_input_output(model, dataloader, int(sys.argv[2]),labelled=False)
elif len(sys.argv) == 2:
    if sys.argv[1] == "export":
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tf_lite_model = converter.convert()
        with open("model.tflite", "wb") as f:
            f.write(tf_lite_model)