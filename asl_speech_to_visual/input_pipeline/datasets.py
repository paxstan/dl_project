import gin
import os
import logging
import pandas as pd
import numpy as np
from glob import glob
import tensorflow_datasets as tfds
import tensorflow as tf
from input_pipeline.preprocessing import preprocess, VectorizeChar, remove_special_characters, extract_all_chars, \
    read_flac_file, path_to_audio

SPEECH_DTYPE = tf.float32
LABEL_DTYPE = tf.int32
AUTOTUNE = tf.data.AUTOTUNE


@gin.configurable()
def load(dataset_name, data_dir):
    train_data, test_data, val_data = [], [], []
    logging.info(f"Preparing dataset {dataset_name}...")

    if dataset_name == "lj_dataset":
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

    elif dataset_name == "libri_speech":
        train_file_path = glob('{}/train-clean-100/**/**/**/*.txt'.format(data_dir))
        val_file_path = glob('{}/dev-clean/**/**/**/*.txt'.format(data_dir))
        test_file_path = glob('{}/test-clean/**/**/**/*.txt'.format(data_dir))
        train_data = pd.DataFrame(map_audio_trans_libri(train_file_path))
        val_data = pd.DataFrame(map_audio_trans_libri(val_file_path))
        test_data = pd.DataFrame(map_audio_trans_libri(test_file_path))

        # (ds_train, ds_val, ds_test), ds_info = tfds.load(
        #     'librispeech',
        #     split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        #     shuffle_files=True,
        #     as_supervised=True,
        #     with_info=True,
        #     data_dir=data_dir
        # )

    return train_data, val_data, test_data, data_dir


def map_audio_trans_libri(file_paths):
    data_list = []
    for file_path in file_paths:
        with open(file_path) as file:
            lines = file.read().split("\n")
            for line in lines:
                try:
                    audio_name, transcript = line.split(" ", 1)
                    audio_path = os.path.join(os.path.split(file_path)[0], "{}.flac".format(audio_name))
                    file_size = os.path.getsize(audio_path)
                    data_list.append(
                        {'wav_filename': audio_path, 'wav_filesize': file_size, 'transcript': transcript})
                except Exception as e:
                    print(e)
    return data_list


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
        data_frame['transcript'] = data_frame['transcript'].map(
            lambda x: vectorizer(x))
        # nan_index = data_frame['transcript'].index[data_frame['transcript'].apply(np.isnan)]
        data_frame.dropna(inplace=True)
        text_list = data_frame['transcript'].tolist()
        data_frame['tensorvalue'] = data_frame['wav_filename'].map(
            lambda x: read_flac_file(x))
        data_frame.dropna(inplace=True)
        audio_list = data_frame['tensorvalue'].tolist()
        text_ds = tf.data.Dataset.from_tensor_slices(text_list)
        audio_ds = tf.data.Dataset.from_tensor_slices(audio_list)
        # tt = audio_ds.take(1)
        # path_to_audio(tt)
        # audio_ds = audio_ds.map(
        #     path_to_audio, num_parallel_calls=tf.data.AUTOTUNE
        # )
        ds = tf.data.Dataset.zip((audio_ds, text_ds))
        ds = ds.map(lambda x, y: {"source": x, "target": y})
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds, vectorizer.get_vocabulary()
