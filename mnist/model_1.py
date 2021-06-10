import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np

# 데이터 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

x_train = x_train / 255.
x_test = x_test / 255.

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

checkpoint_path = "model/checkpoint/mnist.chpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_best_only=True,
                                                         save_weights_only=True,
                                                         verbose=1, monitor='val_loss'
                                                         )

history = model.fit(x_train, y_train, verbose=1, batch_size=32, validation_data=(x_test, y_test), epochs=10,
          callbacks=[checkpoint_callback])

model.load_weights(checkpoint_path)
print("\n")
model.evaluate(x_test, y_test)