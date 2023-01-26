import gin
import os
import logging
import pandas as pd
import numpy as np
from glob import glob
import tensorflow as tf
from input_pipeline.preprocessing import read_flac_file, path_to_audio, fetch_sound_text_mapping, \
    deepspeech_preprocess, PreProcess

SPEECH_DTYPE = tf.float32
LABEL_DTYPE = tf.int32
AUTOTUNE = tf.data.AUTOTUNE
REQUIRED_SAMPLE_RATE = 16000
AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256
BATCH_SIZE = 2
SEED = 42
preprocess = PreProcess()

@gin.configurable()
class LoadDataset(object):
    def __init__(self, data_dir, dataset_name, model_name):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.train_data_frame = pd.DataFrame()
        self.val_data_frame = pd.DataFrame()
        self.test_data_frame = pd.DataFrame()
        self.ds_train = None
        self.ds_val = None
        self.ds_test = None

    def load(self):
        logging.info(f"Preparing dataset {self.dataset_name} for {self.model_name}...")

        if self.dataset_name == "lj_dataset":
            path_csv = os.path.join(self.data_dir, "metadata.csv")
            data_frame = pd.read_csv(path_csv, delimiter="|", names=["ID", "Transcription", "Normalized Transcription"])
            data_frame.columns = ["wav_filename", "wav_filesize", "transcript"]
            data_frame["wav_name"] = data_frame['wav_filename']
            data_frame['wav_filename'] = data_frame['wav_filename'].map(
                lambda x: os.path.join(self.data_dir, "wavs", '{}.wav'.format(x)))
            data_frame['wav_filesize'] = data_frame['wav_filename'].map(
                lambda x: os.path.getsize(x))
            data_frame.dropna(inplace=True)

            # train test split
            train_test_split = np.random.rand(len(data_frame)) < 0.8
            self.train_data_frame = data_frame[train_test_split]
            test_inter_data = data_frame[~train_test_split]

            # dev test split
            dev_test_split = np.random.rand(len(test_inter_data)) <= 0.5
            self.val_data_frame = test_inter_data[dev_test_split]
            self.test_data_frame = test_inter_data[~dev_test_split]

        elif self.dataset_name == "speech_command":

            train_list = []
            # dev data
            dev_file = open(os.path.join(self.data_dir, "validation_list.txt"), "r").read()
            dev_collection = dev_file.split("\n")
            dev_list = []

            # test data
            test_file = open(os.path.join(self.data_dir, "testing_list.txt"), "r").read()
            test_collection = test_file.split("\n")
            test_list = []

            directory = os.fsencode(self.data_dir)

            for subdir in os.listdir(directory):
                subdirectory = os.fsdecode(subdir)
                print(subdirectory)
                subdirectory_path = os.path.join(self.data_dir, subdirectory)
                if not subdirectory.startswith("_background") and os.path.isdir(subdirectory_path):
                    for file in os.listdir(subdirectory_path):
                        audio_file = os.fsdecode(file)
                        path_to_file = os.path.join(subdirectory, audio_file)
                        file_size = os.path.getsize(os.path.join(self.data_dir, path_to_file))
                        if path_to_file in dev_collection:
                            dev_list.append(
                                {"wav_filename": os.path.join(self.data_dir, path_to_file), "wav_name": audio_file,
                                 "wav_filesize": file_size, "transcript": subdirectory})
                        elif path_to_file in test_collection:
                            test_list.append(
                                {"wav_filename": os.path.join(self.data_dir, path_to_file), "wav_name": audio_file,
                                 "wav_filesize": file_size, "transcript": subdirectory})
                        else:
                            train_list.append(
                                {"wav_filename": os.path.join(self.data_dir, path_to_file), "wav_name": audio_file,
                                 "wav_filesize": file_size, "transcript": subdirectory})

            self.train_data_frame = pd.DataFrame(train_list)
            self.val_data_frame = pd.DataFrame(dev_list)
            self.test_data_frame = pd.DataFrame(test_list)

        elif self.dataset_name == "libri_speech":
            train_file_path = glob('{}/train-clean-100/**/**/**/*.txt'.format(self.data_dir))
            val_file_path = glob('{}/dev-clean/**/**/**/*.txt'.format(self.data_dir))
            test_file_path = glob('{}/test-clean/**/**/**/*.txt'.format(self.data_dir))
            self.train_data_frame = pd.DataFrame(map_audio_trans_libri(train_file_path))
            self.val_data_frame = pd.DataFrame(map_audio_trans_libri(val_file_path))
            self.test_data_frame = pd.DataFrame(map_audio_trans_libri(test_file_path))

    def load_data_for_model(self):
        if self.model_name == "deep_speech":
            data_path_dir = os.path.join(self.data_dir, "deep_speech")
            if not os.path.exists(data_path_dir):
                self.load()
                os.makedirs(data_path_dir)
                self.deepspeech_save_data(data_path_dir)
                print("Data created")
            else:
                print("Data already exist!!")
        else:
            tf_record_dir = os.path.join(self.data_dir, "TfRecord")
            if not os.path.exists(tf_record_dir):
                self.load()
                os.makedirs(tf_record_dir)
                train_samples = fetch_sound_text_mapping(self.train_data_frame)
                val_samples = fetch_sound_text_mapping(self.val_data_frame)
                test_samples = fetch_sound_text_mapping(self.test_data_frame)
                create_tf_records("train", train_samples, tf_record_dir)
                create_tf_records("val", test_samples, tf_record_dir)
                create_tf_records("test", val_samples, tf_record_dir)
            self.ds_train = load_tf_record(tf_record_dir, "train")
            self.ds_val = load_tf_record(tf_record_dir, "val")
            self.ds_test = load_tf_record(tf_record_dir, "test")

    def deepspeech_save_data(self, data_path_dir):
        self.train_data_frame = deepspeech_preprocess(self.train_data_frame)
        self.val_data_frame = deepspeech_preprocess(self.val_data_frame)
        self.test_data_frame = deepspeech_preprocess(self.test_data_frame)
        self.train_data_frame.to_csv(os.path.join(data_path_dir, "train.csv"), index=False, encoding='utf-8')
        self.val_data_frame.to_csv(os.path.join(data_path_dir, "dev.csv"), index=False, encoding='utf-8')
        self.test_data_frame.to_csv(os.path.join(data_path_dir, "test.csv"), index=False, encoding='utf-8')


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
                        {'wav_name': audio_name, 'wav_filename': audio_path,
                         'wav_filesize': file_size, 'transcript': transcript})
                except Exception as e:
                    print(e)
    return data_list


# def preprocess_data(model_name, data_frame, batch_size=0):
#     if model_name == "transformer":
#         vectorizer = VectorizeChar()
#         data_frame['transcript'] = data_frame['transcript'].map(
#             lambda x: vectorizer(x))
#         data_frame.dropna(inplace=True)
#         text_list = data_frame['transcript'].tolist()
#         data_frame['tensorvalue'] = data_frame['wav_filename'].map(
#             lambda x: read_flac_file(x))
#         data_frame.dropna(inplace=True)
#         audio_list = data_frame['tensorvalue'].tolist()
#         text_ds = tf.data.Dataset.from_tensor_slices(text_list)
#         audio_ds = tf.data.Dataset.from_tensor_slices(audio_list)
#         # audio_ds = audio_ds.map(
#         #     path_to_audio, num_parallel_calls=tf.data.AUTOTUNE
#         # )
#         ds = tf.data.Dataset.zip((audio_ds, text_ds))
#         ds = ds.map(lambda x, y: {"source": x, "target": y})
#         ds = ds.batch(batch_size)
#         ds = ds.prefetch(tf.data.AUTOTUNE)
#         return ds, vectorizer.get_vocabulary()
#     elif model_name == "wav2vec2":
#         tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="<unk>", pad_token="<pad>")
#         feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
#                                                      do_normalize=True, return_attention_mask=False,
#                                                      do_lower_case=False)
#         processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
#         with processor.as_target_processor():
#             data_frame['labels'] = data_frame['transcript'].map(
#                 lambda x: processor(x).input_ids
#             )
#         data_frame['input_values'] = data_frame['wav_filename'].map(
#             lambda x: read_flac_file(x))
#         data_frame['input_values'] = data_frame['input_values'].map(
#             lambda x: processor(x, sampling_rate=16000).input_values[0])
#         data_frame['input_length'] = data_frame['input_values'].map(
#             lambda x: len(x))
#
#         text_list = data_frame['transcript'].tolist()
#         data_frame['tensorvalue'] = data_frame['wav_filename'].map(
#             lambda x: read_flac_file(x))
#         audio_list = data_frame['tensorvalue'].tolist()
#         text_ds = tf.data.Dataset.from_tensor_slices(text_list)
#         audio_ds = tf.data.Dataset.from_tensor_slices(audio_list)
#         ds = tf.data.Dataset.zip((audio_ds, text_ds))
#         ds = ds.map(lambda x, y: {"source": x, "target": y})
#         ds = ds.batch(batch_size)
#         ds = ds.prefetch(tf.data.AUTOTUNE)


def load_tf_record(tf_record_dir, file_type):
    dataset = tf.data.TFRecordDataset(
        os.path.join(tf_record_dir, "{}.tfrecord-00000-of-00001".format(file_type))).map(
        read_tfrecords, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=(AUDIO_MAXLEN, LABEL_MAXLEN), padding_values=(0.0, 0))
    return dataset.prefetch(tf.data.AUTOTUNE)


# Read the data back out.
def read_tfrecords(record: tf.train.Example):
    desc = {
        "speech": tf.io.FixedLenFeature((), tf.string),
        "label": tf.io.FixedLenFeature((), tf.string),
    }
    record = tf.io.parse_single_example(record, desc)

    speech = tf.io.parse_tensor(record["speech"], out_type=tf.float32)
    label = tf.io.parse_tensor(record["label"], out_type=tf.int32)

    return speech, label


def create_tf_records(file_type, samples, tf_record_dir):
    path_tf_record = os.path.join(tf_record_dir, '{}.tfrecord-00000-of-00001'
                                  .format(file_type))
    with tf.io.TFRecordWriter(path_tf_record) as writer:
        for sample in samples:
            audio_sample = tf.io.serialize_tensor(preprocess.preprocess_speech(sample[0]))
            text_sample = tf.io.serialize_tensor(preprocess.preprocess_text(sample[1]))
            writer.write(
                serialize_example(audio_sample.numpy(), text_sample.numpy()))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(audio, text):
    """
  Creates a tf.train.Example message ready to be written to a file.
  """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'speech': _bytes_feature(audio),
        'label': _bytes_feature(text),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def prepare_dataset(processor, batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch
