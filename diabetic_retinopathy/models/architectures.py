import gin
import tensorflow as tf

from models.layers import vgg_block


@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """

    assert n_blocks > 0, 'Number of blocks has to be at least 1.'

    inputs = tf.keras.Input(input_shape)
    #
    out = tf.keras.applications.densenet.DenseNet121(include_top=False,
                                                     weights='imagenet',
                                                     input_tensor=inputs,
                                                     input_shape=(256, 256, 3),
                                                     pooling='avg',
                                                     classes=2,
                                                     classifier_activation='softmax')
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')


def dense_net_model(input_shape, n_classes):
    inputs = tf.keras.Input(input_shape)
    out = tf.keras.applications.densenet.DenseNet121(include_top=False,
                                                     weights=None,
                                                     input_tensor=inputs,
                                                     input_shape=(256, 256, 3),
                                                     pooling='avg',
                                                     classes=n_classes,
                                                     classifier_activation='softmax')
    return out


def res_net_model(input_shape, n_classes):
    inputs = tf.keras.Input(input_shape)
    out = tf.keras.applications.resnet.ResNet101(include_top=False,
                                                 weights=None,
                                                 input_tensor=inputs,
                                                 input_shape=input_shape,
                                                 pooling='avg',
                                                 classes=n_classes,
                                                 classifier_activation='softmax'
                                                 )
    return out
