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
    out = vgg_block(inputs, base_filters)
    for i in range(2, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='vgg_like')


def dense_net121_model(input_shape, n_classes, dense_units=32, dropout_rate=0.2):
    base_model = tf.keras.applications.densenet.DenseNet121(include_top=False,
                                                            weights='imagenet',
                                                            input_shape=input_shape, )
    base_model.trainable = False

    inputs = tf.keras.Input(input_shape)

    out = base_model(inputs, training=False)

    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.softmax)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def res_net101_model(input_shape, n_classes, dense_units=32, dropout_rate=0.2):
    base_model = tf.keras.applications.resnet.ResNet101(include_top=False,
                                                        weights='imagenet',
                                                        input_shape=input_shape, )
    base_model.trainable = False

    inputs = tf.keras.Input(input_shape)

    out = base_model(inputs, training=False)

    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.softmax)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


@gin.configurable
def xception_model(input_shape, n_classes, dense_units, dropout_rate):
    base_model = tf.keras.applications.Xception(weights="imagenet",
                                                input_shape=input_shape,
                                                include_top=False,
                                                )
    base_model.trainable = False

    inputs = tf.keras.Input(input_shape)

    out = base_model(inputs, training=False)

    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


@gin.configurable
def res_net50_model(input_shape, n_classes, dense_units, dropout_rate):
    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                         weights='imagenet',
                                                         input_shape=input_shape,
                                                         )
    base_model.trainable = False

    inputs = tf.keras.Input(input_shape)

    out = base_model(inputs, training=False)

    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.softmax)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


@gin.configurable
def nas_net(input_shape, n_classes, dense_units=32, dropout_rate=0.2):
    base_model = tf.keras.applications.nasnet.NASNetMobile(include_top=False,
                                                           weights='imagenet',
                                                           input_shape=input_shape, )
    base_model.trainable = False

    inputs = tf.keras.Input(input_shape)

    out = base_model(inputs, training=False)

    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.softmax)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


def efficient_netB4_model(input_shape, n_classes, dense_units=32, dropout_rate=0.2):
    base_model = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False,
                                                            weights='imagenet',
                                                            input_shape=input_shape, )
    base_model.trainable = False

    inputs = tf.keras.Input(input_shape)

    out = base_model(inputs, training=False)

    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.softmax)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

def inceptionv3_model(input_shape, n_classes, dense_units=32, dropout_rate=0.2):
    base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                            weights='imagenet',
                                                            input_shape=input_shape, )
    base_model.trainable = False

    inputs = tf.keras.Input(input_shape)

    out = base_model(inputs, training=False)

    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.softmax)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs)




