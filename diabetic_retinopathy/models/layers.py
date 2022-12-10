import gin
import tensorflow as tf



@gin.configurable
def vgg_block(inputs, filters, kernel_size):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.

    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)

    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(inputs)
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu)(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out


def identity_block(X, f, filters):

    F1, F2, F3 = filters

    X_shortcut = X

    X = tf.keras.layers.Conv2D(filters=F1,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F2,
                               kernel_size=(f, f),
                               strides=(1, 1),
                               padding='same',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F3,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)

    X = tf.keras.layers.Add()([X, X_shortcut])  # SKIP Connection
    X = tf.keras.layers.Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, s=2):


    F1, F2, F3 = filters

    X_shortcut = X

    X = tf.keras.layers.Conv2D(filters=F1,
                               kernel_size=(1, 1),
                               strides=(s, s),
                               padding='valid',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F2,
                               kernel_size=(f, f),
                               strides=(1, 1),
                               padding='same',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    X = tf.keras.layers.Activation('relu')(X)

    X = tf.keras.layers.Conv2D(filters=F3,
                               kernel_size=(1, 1),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)

    X_shortcut = tf.keras.layers.Conv2D(filters=F3,
                                        kernel_size=(1, 1),
                                        strides=(s, s),
                                        padding='valid',
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = tf.keras.layers.BatchNormalization(axis=3)(X_shortcut)

    X = tf.keras.layers.Add()([X, X_shortcut])
    X = tf.keras.layers.Activation('relu')(X)

    return X