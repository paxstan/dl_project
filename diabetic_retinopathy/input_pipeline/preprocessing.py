import gin
import numpy
import tensorflow as tf
import cv2
import numpy as np


@gin.configurable
def preprocess(image, label, img_height, img_width, scale=300):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32)

    # image = np.array(image)

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))
    # print(image)
    # print(type(image))
    # image = numpy.array(image)
    # """Resize and Rescale images"""
    # x = tf.reduce_sum(image[image.shape[0] // 2, :, :], axis=1)
    # x_mean = tf.reduce_mean(x);
    # r = tf.reduce_sum(tf.cast(x > (x_mean / 10)), tf.float32) / 2
    # s = scale * 1.0 / r
    # image = cv2.resize(image, (img_height, img_width), fx=s, fy=s)
    #
    # """Subtract local average color"""
    # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), scale / 30), -4, 128)
    #
    # """clip image to 90% to remove boundary"""
    # # b = np.zeros(image.shape)
    # # cv2.circle(b, (image.shape[1] // 2, image.shape[0] // 2), int(scale * 0.9
    # image = tf.convert_to_tensor(image)
    return image, label


def augment(image, label):
    """Data augmentation"""
    # Rotate 90 degree.
    image = tf.image.rot90(image)
    # Random left or right flip
    image = tf.image.random_flip_left_right(image)
    # Random brightness.
    image = tf.image.random_brightness(
        image, max_delta=0.5)

    return image, label
