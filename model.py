import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential
import numpy as np


class BaseNet(Model):

    def __init__(self):
        super(BaseNet, self).__init__()

        self.input_layer = layers.InputLayer()

        self.data_augmentation_1 = tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal')
        self.data_augmentation_2 = tf.keras.layers.experimental.preprocessing.RandomRotation(0.1)

        self.preprocess_input = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 255)

        self.conv_1 = layers.Conv2D(16, 3, padding='same', activation='relu')
        self.conv_2 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv_3 = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')

        self.maxpool = layers.MaxPooling2D()

        self.flatten = layers.Flatten()

        self.dense_1 = layers.Dense(128, activation='relu')
        self.dense_2 = layers.Dense(20, name='predictions')

        self.dropout = layers.Dropout(0.2)


    def call(self, inputs):
        net = self.input_layer(inputs)
        net = self.data_augmentation_1(net)
        net = self.data_augmentation_2(net)
        net = self.preprocess_input(net)

        net = self.conv_1(net)
        net = self.maxpool(net)
        net = self.conv_2(net)
        net = self.maxpool(net)
        net = self.conv_3(net)
        net = self.maxpool(net)
        net = self.dropout(net)
        net = self.flatten(net)
        net = self.dense_1(net)
        out = self.dense_2(net)
        return out


def create_model():
    net = BaseNet()
    net(tf.random.normal((1, 256, 256, 3)))
    # print(out_test)
    return net

