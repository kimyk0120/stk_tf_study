import numpy as np
import itertools
import tensorflow as tf
import os
from model import create_model



if __name__ == '__main__':

    test_dataset = test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "dataset/Food Classification",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(256, 256))




    model = create_model()
    print(model.summary())

    # 가중치 로드
    model.load_weights('output/model_dump/model_epoch_10.h5')
    print(model.summary())

    # b = len(np.concatenate([i for x, i in test_ds], axis=0))  # 1253
    total_test_ds_cnt = 1253
    correct_cnt = 0

    for idx, (img_batch, label_batch) in enumerate(test_dataset):
        y_true_labels = label_batch
        predictions = model.predict(img_batch)
        for idk, predict in enumerate(predictions):
            premax = np.argmax(predict)
            t_true = int(y_true_labels[idk])
            if premax == t_true :
                correct_cnt += 1

    print("accuracy : ", correct_cnt / total_test_ds_cnt * 100)



    print("fin")