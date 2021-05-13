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

        self.conv1 = layers.Conv2D(16, 3, padding='same', activation='relu')
        self.maxpool1 = layers.MaxPooling2D()
        self.conv2 = layers.Conv2D(32, 3, padding='same', activation='relu')
        self.maxpool2 = layers.MaxPooling2D()
        self.conv3 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.maxpool3 = layers.MaxPooling2D()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(20, name='predictions')

    def call(self, inputs):
        net = self.input_layer(inputs)
        net = self.data_augmentation_1(net)
        net = self.data_augmentation_2(net)
        net = self.preprocess_input(net)
        net = self.conv1(net)
        net = self.maxpool1(net)
        net = self.conv2(net)
        net = self.maxpool2(net)
        net = self.conv3(net)
        net = self.maxpool3(net)
        net = self.flatten(net)
        net = self.dense1(net)
        out = self.dense2(net)
        return out


def create_model():
    net = BaseNet()
    net(tf.random.normal((1, 256, 256, 3)))
    # print(out_test)
    return net

