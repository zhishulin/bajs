from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, AveragePooling2D, GlobalMaxPooling2D, Multiply, \
    Lambda, GlobalAveragePooling2D
from keras.layers.convolutional import MaxPooling2D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.utils import plot_model
import keras.backend.tensorflow_backend as kft
from keras.layers import Multiply

from tensorflow import squeeze

from PIL import Image


def tf_nn_avg_pool(x_input):
    # return tf.nn.avg_pool(x_input, ksize=[1, 1, 1, 3], strides=[1, 1, 1, 3], padding='VALID')
    return tf.reduce_mean(x_input, reduction_indices=[3], keep_dims=True)


def tf_nn_max_pool(x_input):
    # return tf.nn.max_pool(x_input, [1, 1, 1, 3], [1, 1, 1, 3], padding='VALID')
    return tf.reduce_max(x_input, reduction_indices=[3], keep_dims=True)


def Spatial_Domain_attention(x_input):
    with K.name_scope("Spatial_Domain"):
        avg_pool = Lambda(tf_nn_avg_pool, name="channels_avg_pool")(x_input)
        max_pool = Lambda(tf_nn_max_pool, name="channels_max_pool")(x_input)

        concat_max_avg = concatenate([max_pool, avg_pool])  # * x 2
        x = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding="same", use_bias=False)(concat_max_avg)

        x = BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
        x = Activation("sigmoid")(x)
        x = Multiply()([x_input, x])
    return x


def CBAM_Block(nb_classes=1001):
    x_input = Input(shape=(299, 299, 3))

    x = Spatial_Domain_attention(x_input)
    model = Model(inputs=x_input, outputs=x, name="CBAM")
    return model


def expend_dim(x):
    x = tf.expand_dims(x, axis=1)
    x = tf.expand_dims(x, axis=1)
    return x


def Channel_Domain_attention(x_input):
    # 8 * 8 * 1536
    with K.name_scope("Channel_Domain"):
        x = GlobalAveragePooling2D()(x_input)  # 全局平均值池化　１　×　１　×　1536
        # extend direction of x
        x = Lambda(expend_dim)(x)
        x = Conv2D(filters=96, strides=(1, 1), kernel_size=(1, 1), use_bias=True)(x)  # 1 * 1 * 96 缩小32倍
        x = Activation("relu")(x)
        x = Conv2D(filters=1536, strides=(1, 1), use_bias=True, kernel_size=(1, 1))(x)  # 1 * 1 * 1536 还原成原来的形状
        x = Activation("sigmoid")(x)
        x = Multiply()([x, x_input])

    return x


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.80
    kft.set_session(tf.Session(config=config))

    CBAM_model = CBAM_Block()
    print("model finished!")
    plot_model(CBAM_model, 'CBAM.png', show_shapes=True)
