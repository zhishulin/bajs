from PIL import Image
import keras.utils as np_utils
import numpy as np


def load_images(images_path, image_height=299, image_wide=299, num_channels=3):
    images = []
    for file_path in images_path:
        img = Image.open(file_path)
        try:
            if len(img.split()) == 4:
                r, g, b, a = img.split()
                img = Image.merge('RGB', (r, g, b))
                img = img.convert('RGB')
            img = img.resize((image_height, image_wide))
        except Exception as e:
            print("can not change the png to rgb!", e)

        img = np.reshape(img, (image_height, image_wide, num_channels))
        images.append(img)

    images = np.array(images)

    return images


def batch_size_images_generator(x_train, y_train, steps_per_epoch, batch_size, image_height=299, image_wide=299,
                                num_channels=3, num_classes=31):
    while 1:

        try:
            steps_per_epoch == int(x_train.shape[0] / batch_size)
        except Exception as e:
            print("steps_per_epoch is wrong!", e)
        for step in range(steps_per_epoch):
            step_x_train_path = x_train[step]
            step_y_label = y_train[step]
            try:
                step_x_train = load_images(images_path=step_x_train_path, image_height=image_height,
                                           image_wide=image_wide, num_channels=num_channels)
                step_y_label = np_utils.to_categorical(step_y_label, num_classes=num_classes)
            except Exception as e:
                print("error:", e)
                continue

            yield (step_x_train, step_y_label)
