import tqdm as tqdm
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from functools import partial
import itertools
import numpy as np
import time
from model import create_model
import os
import os.path as osp
from pathlib import Path
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt


# loss
def loss(model, x, y):
    y_pred = model(x)
    return loss_object(y_true=y, y_pred=y_pred)

# 미분함수
def train_step(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


if __name__ == '__main__':
    # CONFIG
    ROOT_DIR = Path.cwd()
    OUTPUT_DIR = osp.join(ROOT_DIR, 'output')
    MODEL_DIR = osp.join(OUTPUT_DIR, 'model_dump')
    data_dir = "dataset/Food Classification"

    # HYPER PARAMETER
    BATCH_SIZE = 32
    NUM_EPOCH = 10
    LEARNING_RATE = 0.01
    IMG_SIZE = (256, 256)

    # DATASET load
    train_dataset = train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # data parsing
    class_names = train_dataset.class_names

    # data check
    for image_batch, labels_batch in train_dataset:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[np.argmax(labels[i])])
            plt.axis("off")
    plt.show()

    # 데이터 전처리

    # 이미지 입력 성능 향상
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.cache().shuffle(50).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # 데이터 증강
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    # plt.clf()
    # for image, _ in train_dataset.take(1):
    #     plt.figure(figsize=(10, 10))
    #     first_image = image[0]
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    #         plt.imshow(augmented_image[0] / 255)
    #         plt.axis('off')
    # plt.show()


    # 리스케일링 레이어
    preprocess_input = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1. / 255)

    # load model
    model = create_model()
    print(model.summary())

    # loss 객체 정의
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # 최적화 함수 정의
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # 훈련 루프
    train_loss_results = []
    train_accuracy_results = []

    for epoch in range(NUM_EPOCH):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

        for (img_batch, label_batch) in train_dataset:
            img_batch = data_augmentation(img_batch)
            img_batch = preprocess_input(img_batch)
            loss_value, grads = train_step(model, img_batch, label_batch)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg(loss_value)
            epoch_accuracy(label_batch, model(img_batch))
            print('batch step -- loss : {:.3f}'.format(epoch_loss_avg.result()), ',  accuracy : {:.3%} '.format(epoch_accuracy.result()))

        # epoch 종료
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        print("에포크 {:03d}: 손실: {:.3f}, 정확도: {:.3%}".format(epoch,
                                                           epoch_loss_avg.result(),
                                                           epoch_accuracy.result()))

        if epoch % 5 == 0:
            model.save_weights(
                os.path.join(MODEL_DIR, 'model_epoch_{}.h5'.format(epoch + 1)))



    # validation 활용

    # early stopping

    # 모델 평가




    print("fin")