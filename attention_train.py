from keras.optimizers import SGD
import os
import tensorflow as tf
import pandas as pd
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import keras.backend.tensorflow_backend as kft
from attention_inception_v4 import inception_v4_backbone
from net_param import *
from load_data import load_all_divided_data
from images_generator import batch_size_images_generator
from keras import backend as K
from keras.callbacks import EarlyStopping
from new_loss import loss, categorical_accuracy

# create the evn of this program , it very important
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.80
kft.set_session(tf.Session(config=config))


if __name__ == "__main__":
    att_model = inception_v4_backbone(nb_classes=num_classes)

    if is_load_pre_model:
        att_model.load_weights(os.path.join(save_model_path, save_model_weight_name))

    # it is the optimizers of this model
    sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)

    # custom loss
    def mycrossentropy(y_true, y_pred, e=0.1):
        return (1 - e) * K.categorical_crossentropy(y_pred, y_true) + \
               e * K.categorical_crossentropy(y_pred, K.ones_like(y_pred) / num_classes)

    # many classes
    att_model.compile(optimizer=sgd,
                      loss=mycrossentropy,
                      metrics=['accuracy'])

    # att_model.compile(optimizer=sgd,
    #                   loss=loss,
    #                   metrics=[categorical_accuracy])

    checkpoint = ModelCheckpoint(save_model_path + '/inception_att.h5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    # early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

    callback_lists = [checkpoint, ]  # early_stopping]

    x_train, y_train, steps_per_epoch = load_all_divided_data(train_path, batch_size)
    x_validation, y_validation, steps_per_epoch_validation = load_all_divided_data(validation_path, batch_size)
    x_test, y_test, steps_per_epoch_test = load_all_divided_data(test_path, batch_size)

    print("x train is :", x_train)

    history = att_model.fit_generator(
        generator=batch_size_images_generator(x_train=x_train, y_train=y_train, steps_per_epoch=steps_per_epoch,
                                              batch_size=batch_size, num_classes=num_classes, image_height=image_height,
                                              image_wide=image_wide, num_channels=num_channels),
        steps_per_epoch=steps_per_epoch, epochs=epochs_end, initial_epoch=epochs_start,
        validation_data=batch_size_images_generator(x_train=x_validation, y_train=y_validation,
                                                    steps_per_epoch=steps_per_epoch_validation, batch_size=batch_size,
                                                    num_classes=num_classes, image_height=image_height,
                                                    image_wide=image_wide, num_channels=num_channels),
        validation_steps=steps_per_epoch_validation, shuffle=True, callbacks=callback_lists)

    his_result = {
        'accuracy': history.history['accuracy'],
        'loss': history.history['loss'],
        'val_accuracy': history.history['val_accuracy'],
        'val_loss': history.history['val_loss']
    }
    his_result = pd.DataFrame(his_result)
    his_result.to_csv(os.path.join(save_model_path, save_model_history_name))

    att_model.save_weights(os.path.join(save_model_path, save_model_weight_name))

    loss, accuracy = att_model.evaluate_generator(
        generator=batch_size_images_generator(x_train=x_test, y_train=y_test, steps_per_epoch=steps_per_epoch_test,
                                              batch_size=batch_size, num_classes=num_classes, image_height=image_height,
                                              image_wide=image_wide, num_channels=num_channels),
        steps=steps_per_epoch_test)

    print("loss:", loss)
    print("accurcay:", accuracy)
    print("finished this program!")
