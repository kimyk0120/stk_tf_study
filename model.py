import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential
import numpy as np


class BaseNet(Model):

    def __init__(self):
        super(BaseNet, self).__init__()
        self.preprocess_layer_1 = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 255)
        self.conv_1 = layers.Conv2D(64, 3, padding='same', strides=2, activation='relu')
        self.maxpool_1 = layers.MaxPooling2D()
        self.conv_2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.maxpool_2 = layers.MaxPooling2D()
        self.conv_3 = layers.Conv2D(128, 3, padding='same', strides=2, activation='relu')
        self.maxpool_3 = layers.MaxPooling2D()
        self.conv_4 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.maxpool_4 = layers.MaxPooling2D()
        self.dropout_1 = layers.Dropout(0.5)
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(512, activation='relu')
        self.dense_2 = layers.Dense(20, name='predictions')


    def call(self, inputs):
        net = self.preprocess_layer_1(inputs)
        net = self.conv_1(net)
        net = self.maxpool_1(net)
        net = self.conv_2(net)
        net = self.maxpool_2(net)
        net = self.conv_3(net)
        net = self.maxpool_3(net)
        net = self.conv_4(net)
        net = self.maxpool_4(net)
        net = self.dropout_1(net)
        net = self.flatten(net)
        net = self.dense_1(net)
        out = self.dense_2(net)
        return out


def create_model():
    net = BaseNet()
    net(tf.random.normal((1, 256, 256, 3)))
    # print(out_test)
    return net

