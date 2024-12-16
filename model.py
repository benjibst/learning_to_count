import keras 

class ConvLayerRelu():
    def __init__(self, filters, kernel_size = (3,3), strides = (2,2)):
        self.conv = keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding="same",kernel_initializer="he_normal"
        )
        self.activation = keras.layers.ReLU()
    def __call__(self, x):
        return self.activation(self.conv(x))
class ConvLayerLeakyRelu():
    def __init__(self, filters, kernel_size = (3,3), strides = (2,2)):
        self.conv = keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding="same"
        )
        self.activation = keras.layers.LeakyReLU()
    def __call__(self, x):
        return self.activation(self.conv(x))
class ConvLayer():
    def __init__(self, filters, kernel_size = (3,3), strides = (2,2),activation = "linear"):
        self.conv = keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding="same",activation=activation
        )
    def __call__(self, x):
        return self.conv(x)


class FomoModel:
    convtype = ConvLayer
    def __init__(self,img_sz = (96,96),n_kernels = 4):

        self.conv1 = self.convtype(n_kernels) # 48x48
        self.conv2 = self.convtype(n_kernels*2) # 24x24
        self.conv3 = self.convtype(n_kernels*4) # 12x12
        self.conv4 = self.convtype(n_kernels*8) # 6x6
        self.up1 = keras.layers.Resizing(img_sz[0]//8, img_sz[1]//8)
        self.conv5 = self.convtype(n_kernels*4, strides=(1,1))
        self.concat1 = keras.layers.Concatenate()
        self.up2 = keras.layers.Resizing(img_sz[0]//4, img_sz[1]//4)
        self.conv6 = self.convtype(n_kernels*2, strides=(1,1))
        self.concat2 = keras.layers.Concatenate()
        self.up3 = keras.layers.Resizing(img_sz[0]//2, img_sz[1]//2)
        self.conv7 = self.convtype(n_kernels, strides=(1,1))
        self.concat3 = keras.layers.Concatenate()
        self.conv8 = self.convtype(1, kernel_size=(1,1), strides=(1,1),activation="sigmoid")
    def run(self,x):
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

class CustomModel:
    convtype = ConvLayerRelu
    def __init__(self,**kwargs):
        self.conv11 = self.convtype(16, kernel_size=(3,3),strides=(1,1))
        self.conv12 = self.convtype(16, kernel_size=(3,3),strides=(1,1))
        self.pool1 = keras.layers.AvgPool2D((2,2))
        self.conv21 = self.convtype(32, kernel_size=(3,3),strides=(1,1))
        self.conv22 = self.convtype(32, kernel_size=(3,3),strides=(1,1))
        self.pool2 = keras.layers.AvgPool2D((2,2))
        self.conv31 = self.convtype(64, kernel_size=(3,3),strides=(1,1))
        self.conv32 = self.convtype(64, kernel_size=(3,3),strides=(1,1))
        self.convf = self.convtype(1, kernel_size=(1,1),strides=(1,1))
    def run(self,x):
        c11 = self.conv11(x)
        c12 = self.conv12(c11)
        p1 = self.pool1(c12)
        c21 = self.conv21(p1)
        c22 = self.conv22(c21)
        p2 = self.pool2(c22)
        c31 = self.conv31(p2)
        c32 = self.conv32(c31)
        return self.convf(c32)

class CustomModelInception:
    convtype = ConvLayer
    def __init__(self,**kwargs):
        self.conv13 = self.convtype(8, kernel_size=(3,3),strides=(1,1))
        self.conv15 = self.convtype(8, kernel_size=(5,5),strides=(1,1))
        self.conv17 = self.convtype(8, kernel_size=(7,7),strides=(1,1))
        self.conca1 = keras.layers.Concatenate()
        self.conv23 = self.convtype(16, kernel_size=(3,3),strides=(1,1))
        self.conv25 = self.convtype(16, kernel_size=(5,5),strides=(1,1))
        self.conv27 = self.convtype(16, kernel_size=(7,7),strides=(1,1))
        self.conca2 = keras.layers.Concatenate()
        self.pool1 = keras.layers.MaxPool2D((2,2))
        self.conv33 = self.convtype(16, kernel_size=(3,3),strides=(1,1))
        self.conv35 = self.convtype(16, kernel_size=(5,5),strides=(1,1))
        self.conv37 = self.convtype(16, kernel_size=(7,7),strides=(1,1))
        self.conca3 = keras.layers.Concatenate()
        self.conv43 = self.convtype(32, kernel_size=(3,3),strides=(1,1))
        self.conv45 = self.convtype(32, kernel_size=(5,5),strides=(1,1))
        self.conv47 = self.convtype(32, kernel_size=(7,7),strides=(1,1))
        self.conca4 = keras.layers.Concatenate()
        self.pool2 = keras.layers.AvgPool2D((2,2))
        self.conv31 = self.convtype(32, kernel_size=(3,3),strides=(1,1))
        self.conv32 = self.convtype(64, kernel_size=(3,3),strides=(1,1))
        self.convf = self.convtype(1, kernel_size=(1,1),strides=(1,1))
    def run(self,x):
        c13 = self.conv13(x)
        c15 = self.conv15(x)
        c17 = self.conv17(x)
        c1 = self.conca1([c13,c15,c17])
        c23 = self.conv23(c1)
        c25 = self.conv25(c1)
        c27 = self.conv27(c1)
        c2 = self.conca2([c23,c25,c27])
        p1 = self.pool1(c2)
        c33 = self.conv33(p1)
        c35 = self.conv35(p1)
        c37 = self.conv37(p1)
        c3 = self.conca3([c33,c35,c37])
        c43 = self.conv43(c3)
        c45 = self.conv45(c3)
        c47 = self.conv47(c3)
        c4 = self.conca4([c43,c45,c47])
        p2 = self.pool2(c4)
        c31 = self.conv31(p2)
        c32 = self.conv32(c31)
        return self.convf(c32)


