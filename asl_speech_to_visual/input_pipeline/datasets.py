import gin
import os
import logging
import pandas as pd
import numpy as np
from glob import glob
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
from input_pipeline.preprocessing import VectorizeChar


@gin.configurable()
def load(data_dir):
    wavs = glob("{}/**/*.wav".format(data_dir), recursive=True)

    id_to_text = {}
    val = 0
    with open(os.path.join(data_dir, "metadata.csv"), encoding="utf-8") as f:
        for line in f:
            id = line.strip().split("|")[0]
            text = line.strip().split("|")[2]
            id_to_text[id] = text
            val = val + 1
    max_target_len = 200  # all transcripts in out data are < 200 characters
    data = get_data(wavs, id_to_text, max_target_len)
    vectorizer = VectorizeChar(max_target_len)
    print("vocab size", len(vectorizer.get_vocabulary()))
    split = int(len(data) * 0.99)
    train_data = data[:split]
    test_data = data[split:]
    ds = create_tf_dataset(train_data, vectorizer, bs=64)
    val_ds = create_tf_dataset(test_data, vectorizer, bs=4)
    return ds, val_ds, vectorizer, max_target_len

def get_data(wavs, id_to_text, maxlen=50):
    """ returns mapping of audio paths and transcription texts """
    data = []
    for w in wavs:
        id = w.split("/")[-1].split(".")[0]
        if len(id_to_text[id]) < maxlen:
            data.append({"audio": w, "text": id_to_text[id]})
    return data


def create_text_ds(data, vectorizer):
    texts = [_["text"] for _ in data]
    text_ds = [vectorizer(t) for t in texts]
    text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    return text_ds


def path_to_audio(path):
    # spectrogram using stft
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
    stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / stddevs
    audio_len = tf.shape(x)[0]
    # padding to 10 seconds
    pad_len = 2754
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x


def create_audio_ds(data):
    flist = [_["audio"] for _ in data]
    audio_ds = tf.data.Dataset.from_tensor_slices(flist)
    audio_ds = audio_ds.map(
        path_to_audio, num_parallel_calls=tf.data.AUTOTUNE
    )
    return audio_ds


def create_tf_dataset(data, vectorizer, bs=4):
    audio_ds = create_audio_ds(data)
    text_ds = create_text_ds(data, vectorizer)
    ds = tf.data.Dataset.zip((audio_ds, text_ds))
    ds = ds.map(lambda x, y: {"source": x, "target": y})
    ds = ds.batch(bs)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
