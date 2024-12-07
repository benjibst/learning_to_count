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
            filters, kernel_size, strides=strides, padding="same",
        )
        self.activation = keras.layers.LeakyReLU()
    def __call__(self, x):
        return self.activation(self.conv(x))
class ConvLayer():
    def __init__(self, filters, kernel_size = (3,3), strides = (2,2)):
        self.conv = keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding="same"
        )
    def __call__(self, x):
        return self.conv(x)

convtype = ConvLayer


#conv1 = convtype(4)
#conv2 = convtype(8)
#conv3 = convtype(16)
#conv4 = convtype(32)
#up1 = keras.layers.Resizing(12, 12)
#conv5 = convtype(16, strides=(1,1))
#concat1 = keras.layers.Concatenate()
#up2 = keras.layers.Resizing(24, 24)
#conv6 = convtype(8, strides=(1,1))
#concat2 = keras.layers.Concatenate()
#up3 = keras.layers.Resizing(48, 48)
#conv7 = convtype(4, strides=(1,1))
#concat3 = keras.layers.Concatenate()
#conv8 = convtype(1, kernel_size=(1,1), strides=(1,1))
#
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
#    #c7 = concat3([c1, c7]) 
#    c8 = conv8(c7) 
#    return c8

#96*96*1
conv1 = convtype(4)
#48*48*4
conv2 = convtype(8)
#24*24*8
conv3 = convtype(16)
#12*12*16
conv4 = convtype(32)
#6*6*32
up1 = keras.layers.Resizing(12, 12)
#12*12*32
conv5 = convtype(16, strides=(1,1))
#12*12*16
concat1 = keras.layers.Concatenate()
#12*12*32
up2 = keras.layers.Resizing(24, 24)
#24*24*32
conv6 = convtype(8, strides=(1,1))
#24*24*8
concat2 = keras.layers.Concatenate()
#24*24*16
up3 = keras.layers.Resizing(48, 48)
#48*48*16
conv7 = convtype(4, strides=(1,1))
#48*48*4
concat3 = keras.layers.Concatenate()
#48*48*8
conv8 = convtype(1, kernel_size=(1,1), strides=(1,1))
#48*48*1
def run_model(x):
    c1 = conv1(x)
    c2 = conv2(c1)
    c3 = conv3(c2)
    c4 = conv4(c3)
    u1 = up1(c4)
    c5 = conv5(u1)
    c5 = concat1([c3, c5])
    u2 = up2(c5)
    c6 = conv6(u2)
    c6 = concat2([c2, c6])
    u3 = up3(c6)
    c7 = conv7(u3)
    c7 = concat3([c1, c7])
    c8 = conv8(c7)
    return c8

#conv11 = convtype(8, kernel_size=(3,3),strides=(1,1))
#conv12 = convtype(16, kernel_size=(3,3),strides=(1,1))
#pool1 = keras.layers.AvgPool2D((2,2))
#conv21 = convtype(16, kernel_size=(3,3),strides=(1,1))
#conv22 = convtype(32, kernel_size=(3,3),strides=(1,1))
#pool2 = keras.layers.AvgPool2D((2,2))
#conv31 = convtype(32, kernel_size=(3,3),strides=(1,1))
#conv32 = convtype(64, kernel_size=(3,3),strides=(1,1))
#convf = convtype(1, kernel_size=(1,1),strides=(1,1))
#
#
#def run_model(x):
#    c11 = conv11(x)
#    c12 = conv12(c11)
#    p1 = pool1(c12)
#    c21 = conv21(p1)
#    c22 = conv22(c21)
#    p2 = pool2(c22)
#    c31 = conv31(p2)
#    c32 = conv32(c31)
#    return convf(c32)
    
