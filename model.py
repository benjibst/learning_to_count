import keras 

class ConvLayerRelu():
    def __init__(self, filters, kernel_size = (3,3), strides = (2,2)):
        self.conv = keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding="same"
        )
        self.activation = keras.layers.ReLU()
    def __call__(self, x):
        return self.activation(self.conv(x))

#conv1 = ConvLayerRelu(4)
#conv2 = ConvLayerRelu(8)
#conv3 = ConvLayerRelu(16)
#conv4 = ConvLayerRelu(32)
#up1 = keras.layers.UpSampling2D((2, 2))
#conv5 = ConvLayerRelu(16,strides=(1,1))
#concat1 = keras.layers.Concatenate()
#up2 = keras.layers.UpSampling2D((2, 2))
#conv6 = ConvLayerRelu(8,strides=(1,1))
#concat2 = keras.layers.Concatenate()
#up3 = keras.layers.UpSampling2D((2, 2))
#conv7 = ConvLayerRelu(4,strides=(1,1))
#concat3 = keras.layers.Concatenate()
#conv8 = ConvLayerRelu(1,kernel_size=(1,1),strides=(1,1))

#def run_model(x):
#    c1 = conv1(x)
#    c2 = conv2(c1)
#    c3 = conv3(c2)
#    c4 = conv4(c3)
#    u1 = up1(c4)
#    c5 = conv5(u1)
#    c5 = concat1([c3, c5])
#    u2 = up2(c5)
#    c6 = conv6(u2)
#    c6 = concat2([c2, c6])
#    u3 = up3(c6)
#    c7 = conv7(u3)
#    c7 = concat3([c1, c7])
#    c8 = conv8(c7)
#    return c8

conv11 = ConvLayerRelu(4, kernel_size=(3,3),strides=(1,1))
conv12 = ConvLayerRelu(8, kernel_size=(3,3),strides=(1,1))
pool1 = keras.layers.MaxPooling2D((2,2))
conv21 = ConvLayerRelu(16, kernel_size=(3,3),strides=(1,1))
conv22 = ConvLayerRelu(32, kernel_size=(3,3),strides=(1,1))
pool2 = keras.layers.MaxPooling2D((2,2))
conv31 = ConvLayerRelu(64, kernel_size=(3,3),strides=(1,1))
convf = ConvLayerRelu(1, kernel_size=(1,1),strides=(1,1))


def run_model(x):
    c11 = conv11(x)
    c12 = conv12(c11)
    p1 = pool1(c12)
    c21 = conv21(p1)
    c22 = conv22(c21)
    p2 = pool2(c22)
    c31 = conv31(p2)
    return convf(c31)
    
