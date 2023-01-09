import tensorflow as tf
import numpy as np


def predict_word(model, tokenizer, in_text, string_max_length, vocab_size):
    for i in range(string_max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=string_max_length)
        sequence = tf.keras.utils.to_categorical([sequence], num_classes=vocab_size)[0]
        pred = model.predict(sequence, verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    print("out_text: ", in_text)


def word_for_id(word_id, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == word_id:
            return word
    return None
