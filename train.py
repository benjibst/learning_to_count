
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
from dataloader import PedestrianDataIterator
from model import run_model

img_dir = "images"
loaded = False

if os.path.exists("model.keras"):
    print("Loading model from file")
    loaded = True
    model = keras.models.load_model("model.keras")

else:
    print("Creating new model")
    input = keras.Input(batch_shape=(None, 96, 96, 1))
    output = run_model(input)
    model = keras.Model(inputs=input, outputs=output)


def plot_img_grid(images):
    columns = 2
    rows = len(images) // (columns * 3)
    for i in range(1, columns*3 * rows + 1):
        plt.subplot(rows, columns*3, i)
        plt.imshow(images[i-1])
    plt.show()
def plot_model_input_output(model, data_loader,n):
    if n % 2 != 0:
        n += 1
    images_heatmaps = []
    choose = random.sample(range(len(data_loader)), n)
    for i in choose:
        img, heatmap = data_loader.__getitem__(i)
        images_heatmaps.append(heatmap[0])
        images_heatmaps.append(img[0])
        heatmap_pred = model.predict(img)[0]
        images_heatmaps.append(heatmap_pred)
    plot_img_grid(images_heatmaps)
    plt.show()
heatmap_div = 4
tain_loader = PedestrianDataIterator(image_size=(96,96),labels_file="labels/train.json",heatmap_div=heatmap_div)
val_loader = PedestrianDataIterator(image_size=(96,96),labels_file="labels/val.json",heatmap_div=heatmap_div)
test_loader = PedestrianDataIterator(image_size=(96,96),labels_file="labels/test.json",heatmap_div=heatmap_div)

if len(sys.argv) == 3:
    if sys.argv[1] not in ["train","test"]:
        print("Usage: ")
        print("  python train.py train <epochs>")
        print("  python train.py test <n_images>")

    opt = keras.optimizers.Adam(learning_rate=0.002,weight_decay=0.0001)
    model.compile(optimizer=opt, loss="mse")
    model.summary()

    if sys.argv[1] == "train":
        print(f"Training model for {int(sys.argv[2])} epochs")
        model.fit(test_loader, epochs=int(sys.argv[2]), validation_data=val_loader)
        print("Testing model")
        model.evaluate(test_loader)
        model.save("model.keras")
    else:
        test_loader.batch_size = 1
        plot_model_input_output(model, test_loader, int(sys.argv[2]))
elif len(sys.argv) == 2 and sys.argv[1] == "tflite":
    if not loaded:
        print("First train the model")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = tain_loader.representative_data_gen
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
    os.system("xxd -i model.tflite > model.tflite.h")
        