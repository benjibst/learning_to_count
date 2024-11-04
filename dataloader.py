import keras.backend
import numpy as np
import keras
import json
import os
import matplotlib.pyplot as plt

#make keras use gpu

# Custom DataLoader class inheriting from keras.utils.Sequence
class DataLoaderFactory():
    def __init__(self):
        self.image_paths,self.labels = self.load_labelled_img_paths()
        self.n_labelled = len(self.labels)
        self.unlabelled_image_paths = self.load_unlabelled_img_paths(self.image_paths)

    def load_unlabelled_img_paths(self,labelled_image_paths):
        unlabelled_image_paths = []
        for img in os.listdir("images"):
            if img not in labelled_image_paths:
                unlabelled_image_paths.append("images/"+img)
        return unlabelled_image_paths
    def load_labelled_img_paths(self):
        with open("labels/_labels.json","r") as f:
            labels = json.load(f)
        image_paths = []
        labels_list = []
        for k,v in labels.items():
            image_paths.append("images/"+k)
            labels_list.append(v)
        print(f"Loaded {len(labels_list)} labels")
        return image_paths,labels_list
    
    def get_dataloader(self,n_samples):
        if(n_samples>self.n_labelled):
            n_samples = self.n_labelled
        if(n_samples==0):
            raise ValueError("No labelled data available")
        iterator_images = self.image_paths[0:n_samples]
        iterator_labels = self.labels[0:n_samples]
        self.image_paths = self.image_paths[n_samples:]
        self.labels = self.labels[n_samples:]
        self.n_labelled -= n_samples
        return PedestrianDataIterator(image_paths = iterator_images,labels = iterator_labels)

class PedestrianDataIterator(keras.utils.Sequence):
    def __init__(self, batch_size=8, image_size=(224, 224), max_count=40, image_paths = None,labels=None):
        self.batch_size = batch_size
        self.image_size = image_size
        self.max_count = max_count
        self.image_paths = image_paths
        self.labels = labels
            
    def __len__(self):
        return int(len(self.image_paths) / self.batch_size)

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size: (index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_image_paths, batch_labels)
        return X, y

    def __data_generation(self, batch_image_paths, batch_labels):
        X = np.empty((self.batch_size, *self.image_size, 1))  # Assuming 3-channel RGB images
        y = np.empty((self.batch_size), dtype=int)

        for i, file_path in enumerate(batch_image_paths):
            img = keras.utils.img_to_array(keras.utils.load_img(file_path, target_size=self.image_size,keep_aspect_ratio=True,color_mode="grayscale")) / 255.0  # Normalize the image to [0, 1]
            X[i,] = img
            y[i] = batch_labels[i] / self.max_count

        return X, y


# Example usage:

# Create a custom data loader
data_loader = DataLoaderFactory()
n = data_loader.n_labelled
n_train = int(n*3/4)
n_val = int(n*1/8)
n_test = int(n*1/8)
train_data_loader = data_loader.get_dataloader(n_train)
val_data_loader = data_loader.get_dataloader(n_val)
test_data_loader = data_loader.get_dataloader(n_test)

# Example of how to use the custom data loader with a Keras model
augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomTranslation(0.1,0.1),
])
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(224, 224, 1)),
    augmentation,
    keras.layers.Conv2D(8, (9, 9), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(8, (5, 5), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(1, activation="linear")
])

model.compile(optimizer='adam', loss = "mse", metrics=['accuracy'])

# Train the model using the custom data loader7
model.fit(train_data_loader, epochs=10,validation_data=val_data_loader)
print("Evaluating model")
model.evaluate(test_data_loader)

rand_unlabelled = np.random.choice(data_loader.unlabelled_image_paths,9)
images = []
for image in rand_unlabelled:
    images.append(keras.utils.img_to_array(keras.utils.load_img(image, target_size=(224, 224),keep_aspect_ratio=True,color_mode="grayscale")))
fig,ax = plt.subplots(3,3)
for i in range(3):
    for j in range(3):
        ax[i,j].imshow(images[i*3+j])
        pred = model(np.expand_dims(images[i*3+j]/255.0,0))
        ax[i,j].set_title(f"Prediction: {pred*40}")
plt.show()
