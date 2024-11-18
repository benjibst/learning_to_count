
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
from model import run_model

img_dir = "images"
test_dir = "images_test"
loaded = False

if os.path.exists("model.keras"):
    print("Loading model from file")
    loaded = True
    model = keras.models.load_model("model.keras")

else:
    print("Creating new model")
    model = run_model


def plot_img_grid(images,img_per_col = 3):
    columns = 2
    rows = len(images) // (columns * img_per_col)
    print(f"Plotting {len(images)} images on a {rows}x{columns*img_per_col} grid")
    for i in range(1, columns*img_per_col * rows + 1):
        plt.subplot(rows, columns*img_per_col, i)
        plt.imshow(images[i-1])
    plt.show()
def plot_model_input_output(model, data_loader,n,labelled=True):
    if n % 2 != 0:
        n += 1
    images_heatmaps = []

    choose = random.sample(range(len(data_loader)), n)
    for i in choose:
        if labelled:
            img, heatmap = data_loader.__getitem__(i)
            images_heatmaps.append(heatmap[0])
        else:
            img = data_loader.__getitem__(i)
        images_heatmaps.append(img[0])
        heatmap_pred = model.predict(img)[0]
        images_heatmaps.append(heatmap_pred)
    if(labelled):
        plot_img_grid(images_heatmaps)
    else:
        plot_img_grid(images_heatmaps,2)
    plt.show()

data_loader = DataLoaderFactory(image_size=(96, 96),heatmap_div=4)
n = data_loader.n_labelled / 8
n_train,n_val,n_test = int(n*6),int(n),int(n)
train_data_loader = data_loader.get_dataloader(n_train)
val_data_loader = data_loader.get_dataloader(n_val)
test_data_loader = data_loader.get_dataloader(n_test)
if len(sys.argv) == 3:
    if sys.argv[1] in ("train", "infer"):
        if not loaded:
            input = keras.Input(batch_shape=(None, 96, 96, 1))
            output = run_model(input)
            model = keras.Model(inputs=input, outputs=output)
        opt = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=opt, loss="mse")
        model.summary()
        n = data_loader.n_labelled
        

        if sys.argv[1] == "train":
            print(f"Training model for {int(sys.argv[2])} epochs")
            model.fit(train_data_loader, epochs=int(sys.argv[2]), validation_data=val_data_loader)
            print("Testing model")
            model.evaluate(test_data_loader)
            model.save("model.keras")
        else:
            train_data_loader.batch_size = 1
            plot_model_input_output(model, train_data_loader, int(sys.argv[2]))

    if sys.argv[1] == "test":
        dataloader = UnlabelledDataIterator("images_test",image_size=(96,96))
        plot_model_input_output(model, dataloader, int(sys.argv[2]),labelled=False)
else:
    if loaded:
        data = UnlabelledDataIterator(test_dir,image_size=(96,96),normalize=False)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = data.representative_data_gen
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        tflite_model = converter.convert()
        with open("model.tflite", "wb") as f:
            f.write(tflite_model)
        os.system("xxd -i model.tflite > model.tflite.h")
        