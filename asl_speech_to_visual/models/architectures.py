import tensorflow as tf


def lstm_model(string_max_length, vocab_size):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(string_max_length, vocab_size)))
    model.add(tf.keras.layers.LSTM(units=32))
    model.add(tf.keras.layers.Dense(units=vocab_size, activation='softmax'))
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_lstm.png', show_shapes=True)
    return model
