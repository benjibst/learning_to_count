import keras 

class FomoModel(keras.models.Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            4,
            (3, 3),
            activation="relu",
            padding="same",
            strides=(2, 2),
        )
        self.conv2 = keras.layers.Conv2D(
            8,
            (3, 3),
            activation="relu",
            padding="same",
            strides=(2, 2),
        )
        self.conv3 = keras.layers.Conv2D(
            16,
            (3, 3),
            activation="relu",
            padding="same",
            strides=(2, 2),
        )
        self.conv4 = keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            strides=(2, 2),
        )
        self.up1 = keras.layers.UpSampling2D((2, 2))
        self.conv5 = keras.layers.Conv2D(
            16,
            (3, 3),
            activation="relu",
            padding="same",
            strides=(1, 1),
        )
        self.concat1 = keras.layers.Concatenate()
        self.up2 = keras.layers.UpSampling2D((2, 2))
        self.conv6 = keras.layers.Conv2D(
            8,
            (3, 3),
            activation="relu",
            padding="same",
            strides=(1, 1),
        )
        self.concat2 = keras.layers.Concatenate()
        self.up3 = keras.layers.UpSampling2D((2, 2))
        self.conv7 = keras.layers.Conv2D(
            4,
            (3, 3),
            activation="relu",
            padding="same",
            strides=(1, 1),
        )
        self.concat3 = keras.layers.Concatenate()
        self.conv8 = keras.layers.Conv2D(
            1, 
            (1, 1), 
            activation="relu", 
            padding="same",
            strides=(1, 1))


    def call(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        u1 = self.up1(c4)
        c5 = self.conv5(u1)
        c5 = self.concat1([c3, c5])
        u2 = self.up2(c5)
        c6 = self.conv6(u2)
        c6 = self.concat2([c2, c6])
        u3 = self.up3(c6)
        c7 = self.conv7(u3)
        c7 = self.concat3([c1, c7])
        c8 = self.conv8(c7)
        return c8
