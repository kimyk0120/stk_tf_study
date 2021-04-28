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

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def loss(y_true, y_pred):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return loss


@tf.function
def train_step(img_batch, label_batch, model, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(img_batch)
        y_true = label_batch
        loss_value = loss(y_true, y_pred)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_value


if __name__ == '__main__':
    # define directory
    ROOT_DIR = Path.cwd()
    OUTPUT_DIR = osp.join(ROOT_DIR, 'output')
    MODEL_DIR = osp.join(OUTPUT_DIR, 'model_dump')

    # model config
    batch_size = 16
    num_epoch = 10

    # load model
    logger.info('Create model start')
    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
    logger.info('Create model success')

    # make dataset for train
    logger.info('Make dataset with mnist')

    train_dataset = train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset/Food Classification",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(256, 256),
        batch_size=batch_size)
    # batch_generator = train_dataset.shuffle(40).batch(batch_size)
    logger.info('Make dataset success')

    logger.info('Training start')
    start_time = time.time()
    for epoch in tqdm(range(num_epoch)):
        for idx, (img_batch, label_batch) in enumerate(train_dataset):
            loss_value = train_step(img_batch, label_batch, model, optimizer)
        logger.info('Epoch: {} Time: {:.2}s | Loss: {:.8f}'.format(epoch + 1, time.time() - start_time, min(loss_value).numpy()))
        start_time = time.time()
        model.save_weights(
            os.path.join(MODEL_DIR, 'model_epoch_{}.h5'.format(epoch + 1)))
