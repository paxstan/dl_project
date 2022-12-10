import gin
import tensorflow as tf
from models.layers import vgg_block, identity_block, convolutional_block


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


def ResNet50(input_shape, n_classes, dense_units=32, dropout_rate=0.2):
    X_input = tf.keras.layers.Input(input_shape)

    X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)
    # stage1
    X = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2),
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)
    X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(X)
    # stage2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    # stage3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    # stage4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    # X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    # X = identity_block(X, 3, [512, 512, 2048).], stage = 5, block = 'b' )
    # X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(X)
    X = tf.keras.layers.Dense(dense_units, activation=tf.nn.softmax)(X)
    X = tf.keras.layers.Dropout(dropout_rate)(X)
    X = tf.keras.layers.Dense(n_classes)(X)

    model = tf.keras.Model(inputs=X_input, outputs=X)

    return model
