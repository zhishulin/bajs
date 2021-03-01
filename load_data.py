import random
import os
import numpy as np


def random_reshape(x_train, y_train, steps_per_epoch, batch_size=32):
    x_train_result = []
    y_train_result = []

    residue_train = x_train
    for i in range(steps_per_epoch):
        step_train = random.sample(residue_train, batch_size)
        step_train_labels = []
        residue_train = [item for item in residue_train if item not in step_train]
        for j in range(batch_size):
            label_location = x_train.index(step_train[j])
            step_train_labels.append(y_train[label_location])

        x_train_result.append(step_train)
        y_train_result.append(step_train_labels)

        if len(residue_train) < batch_size:
            break

    x_train_result = np.array(x_train_result)
    y_train_result = np.array(y_train_result)

    return x_train_result, y_train_result


def load_all_divided_data(dataset_path, batch_size=32, type_classes=0):
    x_train = []
    y_train = []

    for root, dirs, files in os.walk(dataset_path):
        if len(files) <= 10:
            continue

        type_classes += 1

        for file in files:
            file_path = os.path.join(root, file)
            x_train.append(file_path)
            y_train.append(type_classes)

    steps_per_epoch = int(len(x_train) / batch_size)
    x_train, y_train = random_reshape(x_train, y_train,
                                      steps_per_epoch=steps_per_epoch,
                                      batch_size=batch_size)

    return x_train, y_train, steps_per_epoch

