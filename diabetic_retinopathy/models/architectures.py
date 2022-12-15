import gin
import tensorflow as tf

@gin.configurable
def res_net50_model(input_shape, n_classes, dense_units, dropout_rate):
    inputs = tf.keras.Input(input_shape)

    # out = base_model(inputs, training=False)
    base_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                         weights='imagenet',
                                                         input_tensor=inputs
                                                         )
    base_model.trainable = False

    out = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(out)

    return tf.keras.Model(inputs=base_model.input, outputs=outputs)


def efficient_netB4_model(input_shape, n_classes, dense_units=32, dropout_rate=0.2):
    inputs = tf.keras.Input(input_shape)
    base_model = tf.keras.applications.efficientnet.EfficientNetB4(include_top=False,
                                                                   weights='imagenet',
                                                                   input_tensor=inputs)
    base_model.trainable = False

    # out = base_model(inputs, training=False)

    out = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(out)

    return tf.keras.Model(inputs=base_model.input, outputs=outputs)


def vgg16_model(input_shape, n_classes, dense_units=32, dropout_rate=0.2):
    inputs = tf.keras.Input(input_shape)
    base_model = tf.keras.applications.vgg16.VGG16(include_top=False,
                                                   weights='imagenet',
                                                   input_tensor=inputs)
    base_model.trainable = False

    # out = base_model(inputs, training=False)

    out = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(out)

    return tf.keras.Model(inputs=base_model.input, outputs=outputs)
