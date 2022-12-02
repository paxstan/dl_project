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
    # image = tf.image.resize(image, size=(img_height, img_width), preserve_aspect_ratio=True)
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
    blur_image = gaussian_blur(image)
    image = tensorflow_add_weighted(image, blur_image)
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


def tensorflow_add_weighted(img1, img2):
    img = img1 * tf.multiply(tf.ones(tf.shape(img1), dtype=tf.float32), 4) + img2 * tf.multiply(
        tf.ones(tf.shape(img2), dtype=tf.float32), -4)
    return img


def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(img):
    blur = _gaussian_kernel(3, 2, 3, img.dtype)
    img = tf.nn.depthwise_conv2d(img[None], blur, [1, 1, 1, 1], 'SAME')
    return img[0]


def gaussian_blur(img, ksize=5, sigma=1):
    def gaussian_kernel(size=3, sigma=1):
        x_range = tf.range(-(size - 1) // 2, (size - 1) // 2 + 1, 1)
        y_range = tf.range((size - 1) // 2, -(size - 1) // 2 - 1, -1)

        xs, ys = tf.meshgrid(x_range, y_range)
        kernel = tf.exp(-(xs ** 2 + ys ** 2) / (2 * (sigma ** 2))) / (2 * np.pi * (sigma ** 2))
        return tf.cast(kernel / tf.reduce_sum(kernel), tf.float32)

    kernel = gaussian_kernel(ksize, sigma)
    kernel = tf.expand_dims(tf.expand_dims(kernel, axis=-1), axis=-1)
    r, g, b = tf.split(img, [1, 1, 1], axis=-1)
    r = tf.expand_dims(r, axis=0)
    g = tf.expand_dims(g, axis=0)
    b = tf.expand_dims(b, axis=0)
    r_blur = tf.nn.conv2d(r, kernel, [1, 1, 1, 1], 'SAME')
    g_blur = tf.nn.conv2d(g, kernel, [1, 1, 1, 1], 'SAME')
    b_blur = tf.nn.conv2d(b, kernel, [1, 1, 1, 1], 'SAME')
    blur_image = tf.concat([r_blur, g_blur, b_blur], axis=-1)
    return tf.squeeze(blur_image, axis=0)

@gin.configurable
def scale_radius(image, img_height, img_width, scale):
    x = image[image.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    image = cv2.resize(image, (img_height, img_width), fx=s, fy=s)
    return image

# data = tf.data.Dataset.from_tensor_slices(
#     (tf.pad(tf.ones((1, 1, 1, 2)), ((0, 0), (1, 1), (1, 1), (0, 1))), tf.ones((1, 3, 3, 1)))
# ).map(lambda x, y: (apply_blur(x), y)).repeat().batch(10)
