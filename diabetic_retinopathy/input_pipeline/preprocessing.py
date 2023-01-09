import gin
import tensorflow as tf
import cv2
import numpy as np


@gin.configurable
def preprocess(image, scale):
    """Dataset preprocessing"""
    image = cv2.resize(image, (256, 256))

    # scale image to a radius
    x = image[image.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    image_radius = cv2.resize(image, (0, 0), fx=s, fy=s)
    image = image_radius

    # subtract local mean color
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), scale / 30), -4, 128)

    # remove outer 10%
    b = np.zeros(image.shape)
    cv2.circle(b, (image.shape[1] // 2, image.shape[0] // 2), int(scale * 0.9), (1, 1, 1), -1, 8, 0)
    image = image * b + 128 * (1 - b)
    image = cv2.resize(image, (256, 256))

    return image


def augment(image, label):
    """Data augmentation"""

    # Random left or right flip
    image = tf.image.random_flip_left_right(image)
    # Random brightness.
    image = tf.image.random_brightness(image, max_delta=0.1)
    # Random saturation
    image = tf.image.random_saturation(image, lower=0.75, upper=1.5)
    # Random hue
    image = tf.image.random_hue(image, max_delta=0.15)
    # Random contrast
    image = tf.image.random_contrast(image, lower=0.75, upper=1.5)

    return image, label

