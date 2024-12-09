
import os
import random
import sys


# keras backend jax
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import keras
import matplotlib.pyplot as plt
from dataloader import DataIterator
from model import FomoModel as model_imp

img_dir = "/home/benni/dev/learning_to_count_data/images"
labels_dir = "/home/benni/dev/learning_to_count_data/labels"
test_dir = "/home/benni/dev/learning_to_count_data/test"
loaded = False
input_sz = (96, 96)
output_sz = None
if os.path.exists("model.keras"):
    print("Loading model from file")
    loaded = True
    model = keras.models.load_model("model.keras")
    inputs = keras.Input(batch_shape=(None, *input_sz, 1))
    output = model(inputs)
    output_sz = output.shape[1:]
else:
    print("Creating new model")
    input = keras.Input(batch_shape=(None, *input_sz, 1))
    mod = model_imp(img_sz=input_sz,n_kernels=32)
    output = mod.run(input)
    model = keras.Model(inputs=input, outputs=output)
    output_sz = output.shape[1:]


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
        images_heatmaps.append(img[0])
        heatmap_pred = model.predict(img)[0]
        images_heatmaps.append(heatmap[0])
        images_heatmaps.append(heatmap_pred)
    plot_img_grid(images_heatmaps)
    plt.show()
def plot_model_output(model, data_loader,n):
    if n % 2 != 0:
        n += 1
    images_heatmaps = []
    choose = random.sample(range(len(data_loader)), n)
    for i in choose:
        img = data_loader.__getitem__(i)
        images_heatmaps.append(img[0])
        heatmap_pred = model.predict(img)[0]
        images_heatmaps.append(heatmap_pred)
    plot_img_grid(images_heatmaps)
    plt.show()

heatmap_div = input_sz[0] / output_sz[0]
train_loader = DataIterator(f"{labels_dir}/train.json",img_dir,heatmap_div=heatmap_div,image_size=input_sz)
val_loader = DataIterator(f"{labels_dir}/val.json",img_dir,heatmap_div=heatmap_div,image_size=input_sz)
test_loader = DataIterator(f"{labels_dir}/test.json",img_dir,heatmap_div=heatmap_div,image_size=input_sz)
nyctest = DataIterator(None,test_dir,heatmap_div=heatmap_div,image_size=input_sz)

if len(sys.argv) == 3:
    if sys.argv[1] not in ["train","test"]:
        print("Usage: ")
        print("  python train.py train <epochs>")
        print("  python train.py test <n_images>")

    opt = keras.optimizers.Adam(learning_rate=0.005,weight_decay=0.0)
    model.compile(optimizer=opt, loss="mse")
    model.summary()

    if sys.argv[1] == "train":
        print(f"Training model for {int(sys.argv[2])} epochs")
        model.fit(train_loader, epochs=int(sys.argv[2]), validation_data=val_loader)
        print("Testing model")
        model.evaluate(test_loader)
        model.save("model.keras")
    else:
        test_loader.batch_size = 1
        nyctest.batch_size = 1
        #plot_model_output(model, nyctest, int(sys.argv[2]))
        plot_model_input_output(model, test_loader, int(sys.argv[2]))

elif len(sys.argv) == 2 and sys.argv[1] == "tflite":
    if not loaded:
        print("First train the model")
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = test_loader.representative_data_gen
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
    os.system("xxd -i model.tflite > model.tflite.h")
        