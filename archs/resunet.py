## Implementation of ResUNet architecture.
## Adapted from https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model

def batchnorm_relu(inputs):
    """ Batch Normalization & ReLU """
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    return x

def residual_block(inputs, num_filters, strides=1):
    """ Convolutional Layers """
    x = batchnorm_relu(inputs)
    x = Conv2D(num_filters, 3, padding="same", strides=strides, kernel_initializer="he_normal")(x)
    x = batchnorm_relu(x)
    x = Conv2D(num_filters, 3, padding="same", strides=1, kernel_initializer="he_normal")(x)

    """ Shortcut Connection (Identity Mapping) """
    s = Conv2D(num_filters, 1, padding="same", strides=strides, kernel_initializer="he_normal")(inputs)

    """ Addition """
    x = x + s
    return x

def decoder_block(inputs, skip_features, num_filters):
    """ Decoder Block """

    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip_features])
    x = residual_block(x, num_filters, strides=1)
    return x

def ResUNet(input_filters, height, width, n_channels):
    """ RESUNET Architecture """

    inputs = Input((height, width, n_channels))
    filters = input_filters

    """ Endoder 1 """
    x = Conv2D(64, 3, padding="same", strides=1, kernel_initializer="he_normal")(inputs)
    x = batchnorm_relu(x)
    x = Conv2D(64, 3, padding="same", strides=1, kernel_initializer="he_normal")(x)
    s = Conv2D(64, 1, padding="same", kernel_initializer="he_normal")(inputs)
    s1 = x + s

    """ Encoder 2, 3 """
    s2 = residual_block(s1, 128, strides=2)
    s3 = residual_block(s2, 256, strides=2)

    """ Bridge """
    b = residual_block(s3, 512, strides=2)

    """ Decoder 1, 2, 3 """
    x = decoder_block(b, s3, 256)
    x = decoder_block(x, s2, 128)
    x = decoder_block(x, s1, 64)

    """ Classifier """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    """ Model """
    model = Model(inputs, outputs, name="RESUNET")

    return model

# if __name__ == "__main__":
#     shape = (256, 256, 3)
#     model = ResUNet(input_filters=16, height=256, width=256, n_channels=3)
#     model.summary()