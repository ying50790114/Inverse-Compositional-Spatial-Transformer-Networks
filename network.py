from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras import Model
import tensorflow as tf


class Geometric_Predictor(Model):
    def __init__(self, warpDim):
        super(Geometric_Predictor, self).__init__()

        self.conv1 = Conv2D(4,
                            kernel_size=(7, 7),
                            strides=(1, 1),
                            padding='valid',
                            activation='relu',
                            name='conv1',
                            trainable=True)
        self.conv2 = Conv2D(8,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='valid',
                            activation='relu',
                            name='conv2',
                            trainable=True)
        self.pool = MaxPool2D(pool_size=(2, 2),
                              strides=(1, 1),
                              padding='valid',
                              name='pool')
        self.flat = Flatten()

        self.fc0 = Dense(48, activation='relu', name='fc0', trainable=True)

        self.fc1 = Dense(warpDim, name='fc1', trainable=True)

    def call(self, inputs):
        z = self.conv1(inputs)
        z = self.conv2(z)
        z = self.pool(z)
        z = self.flat(z)
        z = self.fc0(z)
        dp = self.fc1(z)
        return dp

class Classifier(Model):
    def __init__(self, labelN):
        super(Classifier, self).__init__()
        self.conv = Conv2D(3,
                            kernel_size=(9, 9),
                            strides=(1, 1),
                            padding='valid',
                            activation='relu',
                            name='conv',
                            trainable=True)

        self.flat = Flatten()

        self.fc = Dense(labelN,
                        name='fc',
                        activation='softmax',
                        trainable=True)

    def call(self, inputs):
        z = self.conv(inputs)
        z = self.flat(z)
        prob = self.fc(z)
        return prob

if __name__ == '__main__':
    import warp
    warp = warp.Process(imgSize='28x28', batchSize=1)

    img = tf.ones([1, 28, 28, 1])
    init_p = warp.genPerturbations(img.get_shape()[0])

    Geometric_Predictor = Geometric_Predictor(warpDim=6)  # affine
    Classifier = Classifier(labelN=10)

    dp = Geometric_Predictor.call(img)
    p = warp.compose(init_p, dp)
    pMtrx = warp.vec2mtrx(p)
    imageWarp = warp.transformImage(img, pMtrx)
    predictions = Classifier.call(imageWarp)
    print('success:', dp.get_shape(), predictions.get_shape())



