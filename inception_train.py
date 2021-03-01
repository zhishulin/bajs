from keras.optimizers import SGD
import tensorflow as tf
import keras.utils as np_utils
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import keras.backend.tensorflow_backend as kft
from inception_v4 import inception_v4_backbone
from net_param import *
from load_data import load_all_divided_data
from images_generator import *
from keras.models import load_model
from keras import backend as K
from keras.callbacks import EarlyStopping


def main(is_load_pre_model=0):

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.80
    kft.set_session(tf.Session(config=config))

    if is_load_pre_model:
        inception_v4 = load_model(os.path.join(save_model_path, save_model_name))
    else:
        inception_v4 = inception_v4_backbone(nb_classes=num_classes)

    sgd = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)

    # custom loss
    # def mycrossentropy(y_true, y_pred, e=0.1):
    #     return (1 - e) * K.categorical_crossentropy(y_pred, y_true) + \
    #            e * K.categorical_crossentropy(y_pred, K.ones_like(y_pred) / num_classes)
    inception_v4.compile(optimizer=sgd,
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

    tensorboad = TensorBoard(log_dir=inception_log_dir)

    checkpoint = ModelCheckpoint(save_model_path + '/inception.h5', monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    # early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1)

    callback_lists = [tensorboad, checkpoint] #, early_stopping]

    x_train, y_train, steps_per_epoch = load_all_divided_data(train_path, batch_size)
    x_validation, y_validation, steps_per_epoch_validation = load_all_divided_data(validation_path, batch_size)
    x_test, y_test, steps_per_epoch_test = load_all_divided_data(test_path, batch_size)

    inception_v4.fit_generator(
        generator=batch_size_images_generator(x_train=x_train, y_train=y_train, steps_per_epoch=steps_per_epoch,
                                              batch_size=batch_size, num_classes=num_classes, image_height=image_height,
                                              image_wide=image_wide, num_channels=num_channels),
        steps_per_epoch=steps_per_epoch, epochs=epochs_end, initial_epoch=epochs_start,
        validation_data=batch_size_images_generator(x_train=x_validation, y_train=y_validation,
                                                    steps_per_epoch=steps_per_epoch_validation, batch_size=batch_size,
                                                    num_classes=num_classes, image_height=image_height,
                                                    image_wide=image_wide, num_channels=num_channels),
        validation_steps=steps_per_epoch_validation, shuffle=False, callbacks=callback_lists)

    inception_v4.save(os.path.join(save_model_path, save_model_name))

    loss, accuracy = inception_v4.evaluate_generator(
        generator=batch_size_images_generator(x_train=x_test, y_train=y_test, steps_per_epoch=steps_per_epoch_test,
                                              batch_size=batch_size, num_classes=num_classes, image_height=image_height,
                                              image_wide=image_wide, num_channels=num_channels),
        steps=steps_per_epoch_test, verbose=1)

    print("loss:", loss)
    print("accurcay:", accuracy)
    print("finished this program!")
    print("finished this program!")


if __name__ == "__main__":

    main(is_load_pre_model)

