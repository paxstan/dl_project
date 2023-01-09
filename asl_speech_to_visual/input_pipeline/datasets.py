import gin
import os
import logging
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds


def load():
    path_csv = "/home/paxstan/Documents/Uni/DL_Lab/dl-lab-22w-team07/asl_speech_to_visual/Data/LJSpeech-1.1/metadata.csv"
    data_frame = pd.read_csv(path_csv, delimiter="|", names=["ID", "Transcription", "Normalized Transcription"])
    nan_index = list(data_frame[data_frame.isnull().any(axis=1)].index.values)
    data_frame.dropna(inplace=True)
    return word_tokenizer(data_frame["Normalized Transcription"].tolist())


def word_tokenizer(sentence_list):
    string_max_length = len(max(sentence_list, key=len))
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=2000, oov_token="<rare>")
    tokenizer.fit_on_texts(sentence_list)
    sequences = tokenizer.texts_to_sequences(sentence_list)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    x, y = list(), list()
    for i, seq in enumerate(sequences):
        for j in range(1, len(seq)):
            in_seq, out_seq = seq[:j], seq[j]
            in_seq = tf.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=string_max_length)[0]
            in_seq = tf.keras.utils.to_categorical([in_seq], num_classes=vocab_size)[0]
            out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
            x.append(in_seq)
            y.append(out_seq)
    print('Total Sequences:', len(x))
    return tokenizer, string_max_length, vocab_size, x, y
