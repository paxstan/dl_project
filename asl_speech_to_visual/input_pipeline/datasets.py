import gin
import os
import logging
import pandas as pd
import numpy as np
from glob import glob
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
from input_pipeline.preprocessing import preprocess, VectorizeChar


@gin.configurable()
def load(dataset_name, data_dir):
    train_data, test_data, val_data = [], [], []

    if dataset_name == "lj_dataset":
        logging.info(f"Preparing dataset {dataset_name}...")
        path_csv = os.path.join(data_dir, "metadata.csv")
        data_frame = pd.read_csv(path_csv, delimiter="|", names=["ID", "Transcription", "Normalized Transcription"])
        # nan_index = list(data_frame[data_frame.isnull().any(axis=1)].index.values)
        data_frame.columns = ["wav_filename", "wav_filesize", "transcript"]
        data_frame['wav_filename'] = data_frame['wav_filename'].map(
            lambda x: os.path.join(data_dir, "wavs", '{}.wav'.format(x)))
        data_frame['wav_filesize'] = data_frame['wav_filename'].map(
            lambda x: os.path.getsize(x))
        data_frame.dropna(inplace=True)

        # train test split
        train_test_split = np.random.rand(len(data_frame)) < 0.8
        train_data = data_frame[train_test_split]
        test_inter_data = data_frame[~train_test_split]

        # dev test split
        dev_test_split = np.random.rand(len(test_inter_data)) <= 0.5
        val_data = test_inter_data[dev_test_split]
        test_data = test_inter_data[~dev_test_split]

    elif dataset_name == "speech_command":
        logging.info(f"Preparing dataset {dataset_name}...")

        train_list = []
        # dev data
        dev_file = open(os.path.join(data_dir, "validation_list.txt"), "r").read()
        dev_collection = dev_file.split("\n")
        dev_list = []

        # test data
        test_file = open(os.path.join(data_dir, "testing_list.txt"), "r").read()
        test_collection = test_file.split("\n")
        test_list = []

        directory = os.fsencode(data_dir)

        for subdir in os.listdir(directory):
            subdirectory = os.fsdecode(subdir)
            print(subdirectory)
            subdirectory_path = os.path.join(data_dir, subdirectory)
            if not subdirectory.startswith("_background") and os.path.isdir(subdirectory_path):
                for file in os.listdir(subdirectory_path):
                    audio_file = os.fsdecode(file)
                    path_to_file = os.path.join(subdirectory, audio_file)
                    file_size = os.path.getsize(os.path.join(data_dir, path_to_file))
                    if path_to_file in dev_collection:
                        dev_list.append(
                            {"wav_filename": os.path.join(data_dir, path_to_file), "path": path_to_file,
                             "wav_filesize": file_size, "transcript": subdirectory})
                    elif path_to_file in test_collection:
                        test_list.append(
                            {"wav_filename": os.path.join(data_dir, path_to_file), "path": path_to_file,
                             "wav_filesize": file_size, "transcript": subdirectory})
                    else:
                        train_list.append(
                            {"wav_filename": os.path.join(data_dir, path_to_file), "path": path_to_file,
                             "wav_filesize": file_size, "transcript": subdirectory})

        train_data = pd.DataFrame(train_list)
        val_data = pd.DataFrame(dev_list)
        test_data = pd.DataFrame(test_list)

    return train_data, val_data, test_data, data_dir


def deepspeech_save_data(data_dir, train_data, val_data, test_data):
    train_data = deepspeech_preprocess(train_data)
    val_data = deepspeech_preprocess(val_data)
    test_data = deepspeech_preprocess(test_data)
    train_data.to_csv(os.path.join(data_dir, "train.csv"), index=False, encoding='utf-8')
    val_data.to_csv(os.path.join(data_dir, "dev.csv"), index=False, encoding='utf-8')
    test_data.to_csv(os.path.join(data_dir, "test.csv"), index=False, encoding='utf-8')


def deepspeech_preprocess(data_frame):
    data_frame = data_frame.apply(preprocess, axis=1)
    data_frame.dropna(inplace=True)
    data_frame.drop('wav_filename', axis=1, inplace=True)
    data_frame.rename(columns={'path': 'wav_filename'}, inplace=True)
    return data_frame


def preprocess_data(model_name, data_frame, batch_size=0):
    if model_name == "transformer":
        vectorizer = VectorizeChar()
        text_list = data_frame['transcript'].map(
            lambda x: vectorizer(x)).tolist()
        audio_list = data_frame['wav_filename'].tolist()
        text_ds = tf.data.Dataset.from_tensor_slices(text_list)
        audio_ds = tf.data.Dataset.from_tensor_slices(audio_list)
        audio_ds = audio_ds.map(
            path_to_audio, num_parallel_calls=tf.data.AUTOTUNE
        )
        ds = tf.data.Dataset.zip((audio_ds, text_ds))
        ds = ds.map(lambda x, y: {"source": x, "target": y})
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds, vectorizer.get_vocabulary()


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
