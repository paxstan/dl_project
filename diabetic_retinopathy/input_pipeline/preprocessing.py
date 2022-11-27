import gin
import tensorflow as tf
import cv2

@gin.configurable
def preprocess(image, label, img_height, img_width, scale=300):
    """Dataset preprocessing: Normalizing and resizing"""

    # Normalize image: `uint8` -> `float32`.
    tf.cast(image, tf.float32) / 255.

    # Resize image
    image = tf.image.resize(image, size=(img_height, img_width))

    """Subtract local average color"""
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), scale / 30), -4, 128)

    """clip image to 90% to remove boundary"""
    # b = np.zeros(image.shape)
    # cv2.circle(b, (image.shape[1] // 2, image.shape[0] // 2), int(scale * 0.9

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