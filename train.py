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

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_loss_avg.update_state(loss_value)
    train_accuracy_metric.update_state(y, logits)

    return loss_value


@tf.function
def test_step(x, y):
    val_logits = model(x, training=False)
    val_loss_value = loss_fn(y, val_logits)

    val_loss_avg.update_state(val_loss_value)
    val_accuracy_metric.update_state(y, val_logits)


if __name__ == '__main__':
    # CONFIG
    ROOT_DIR = Path.cwd()
    OUTPUT_DIR = osp.join(ROOT_DIR, 'output')
    MODEL_DIR = osp.join(OUTPUT_DIR, 'model_dump')
    DATA_DIR = "dataset/Food Classification"

    # HYPER PARAM
    SEED = 123
    SHUFFLE_SIZE = 20
    BATCH_SIZE = 32
    NUM_EPOCH = 100
    LEARNING_RATE = 0.0005  # 1e-3 or 5e-4
    IMG_SIZE = (256, 256)
    INPUT_SHAPE = (1, 256, 256, 3)
    VALIDATION_SPLIT = 0.3

    # DATASET load
    train_dataset = train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"  # label_mode = "categorical" => 레이블을 one-hot vector 화 해서 return
    )


    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    # data parsing
    class_names = train_dataset.class_names

    # data check
    for image_batch, labels_batch in train_dataset:
        print("image_batch.shape : ", image_batch.shape)
        print("labels_batch.shape : ", labels_batch.shape)
        break

    # 이미지 입력 성능 향상
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.cache().shuffle(SHUFFLE_SIZE).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # load model
    model = create_model()
    print(model.summary())

    # loss 객체 정의
    # 정수로; 된; label을; 주면; 내부적으로; one - hot; vector로; 변환해서; 알아서; loss를; 계산=>; SparseCategoricalCrossentrop
    # ont-hot vector = CategoricalCrossentropy
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # 최적화 함수 정의
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # metrics
    train_loss_avg = tf.keras.metrics.Mean()
    train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    val_loss_avg = tf.keras.metrics.Mean()
    val_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()

    # 가시화 데이터 배열
    train_loss_results = []
    train_accuracy_results = []

    val_loss_results = []
    val_accuracy_results = []

    # 훈련 루프
    start_time = time.time()
    print("train start : ", start_time)
    for epoch in range(NUM_EPOCH):

        for batch_idx, (img_batch, label_batch) in enumerate(train_dataset):
            loss_value = train_step(img_batch, label_batch)
            # print(batch_idx + 1, 'training batch step -- loss : {:.3f}'.format(train_loss_avg.result()),
            #       ',  accuracy : {:.3%} '.format(train_accuracy_metric.result()))

        # 평가 루프
        for x_batch_val, y_batch_val in validation_dataset:
            test_step(x_batch_val, y_batch_val)

        # epoch 종료
        print("에포크 {}: ".format(epoch+1), ' / train loss : {:.3f}'.format(train_loss_avg.result()),
              ' / train acc : {:.3%}'.format(train_accuracy_metric.result()),
              ' / val loss : {:.3f}'.format(val_loss_avg.result()),
              ' / val acc : {:.3%}'.format(val_accuracy_metric.result()))

        # 가시화 데이터 저장
        train_loss_results.append(train_loss_avg.result())
        train_accuracy_results.append(train_accuracy_metric.result())

        val_loss_results.append(val_loss_avg.result())
        val_accuracy_results.append(val_accuracy_metric.result())

        # Reset training metrics at the end of each epoch
        train_loss_avg.reset_states()
        train_accuracy_metric.reset_states()
        val_loss_avg.reset_states()
        val_accuracy_metric.reset_states()

        if (epoch+1) % 5 == 0:
            model.save_weights(
                os.path.join(MODEL_DIR, 'model_epoch_{}.h5'.format(epoch + 1)))
            print("save_weights epoch {}".format(epoch + 1))

    end_time = time.time() - start_time
    print("train fin : ", end_time)

    # train, validation 가시화
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    # Summarize history for accuracy
    ax[0].plot(train_accuracy_results)
    ax[0].plot(val_accuracy_results)
    ax[0].set_title('Accuracy vs Epoch')
    ax[0].set(xlabel='epoch', ylabel='accuracy')
    ax[0].legend(['train', 'val'], loc='upper left')

    # Summarize history for loss
    ax[1].plot(train_loss_results)
    ax[1].plot(val_loss_results)
    ax[1].set_title('Loss vs Epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    ax[1].legend(['train', 'val'], loc='upper left')

    # Display plots
    plt.show()



    # 모델 평가


    '''
    참고 
    https://www.kaggle.com/l33tc0d3r/indian-food-classification
    https://www.kaggle.com/itokianarafidinarivo/tutorial-keras-image-classification
    https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch?hl=ko
    
    '''



    print("prcs fin")