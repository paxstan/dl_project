import gin
import os
from glob import glob
import soundfile as sf
import pandas as pd
import torch
from tfrecord.torch.dataset import TFRecordDataset
from input_pipeline.preprocessing import PreProcess, PreProcessWav2Vec2
from datasets import Dataset, Features, Audio, load_dataset

REQUIRED_SAMPLE_RATE = 16000
AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256
BATCH_SIZE = 2
SEED = 42


@gin.configurable()
class LoadDataset(object):
    def __init__(self, data_dir, processor):
        self.data_dir = data_dir
        self.processor = processor
        self.tf_record_dir = os.path.join(data_dir, "TfRecord")
        self.train_path_record = os.path.join(self.tf_record_dir, 'librispeech-train-00000-of-00001.parquet')
        self.val_path_record = os.path.join(self.tf_record_dir, 'librispeech-val-00000-of-00001.parquet')
        self.test_path_record = os.path.join(self.tf_record_dir, 'librispeech-test-00000-of-00001.parquet')

    def load(self):
        if not os.path.exists(self.tf_record_dir):
            os.makedirs(self.tf_record_dir)
            train_text_path = glob('{}/train-clean-100/**/**/**/*.txt'.format(self.data_dir))
            val_text_path = glob('{}/dev-clean/**/**/**/*.txt'.format(self.data_dir))
            test_text_path = glob('{}/test-clean/**/**/**/*.txt'.format(self.data_dir))

            train_flac_path = glob('{}/train-clean-100/**/**/**/*.flac'.format(self.data_dir))
            val_flac_path = glob('{}/dev-clean/**/**/**/*.flac'.format(self.data_dir))
            test_flac_path = glob('{}/test-clean/**/**/**/*.flac'.format(self.data_dir))

            create_parquet_record(self.train_path_record, train_text_path, train_flac_path)
            create_parquet_record(self.val_path_record, val_text_path, val_flac_path)
            create_parquet_record(self.test_path_record, test_text_path, test_flac_path)

        dataset = load_dataset("parquet",
                               data_files={
                                   'train': self.train_path_record,
                                   'test': self.test_path_record,
                                   'val': self.val_path_record})

        return dataset

    def prepare_dataset(self, batch):
        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = self.processor(batch["audio"], sampling_rate=REQUIRED_SAMPLE_RATE).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["text"]).input_ids
        return batch


@gin.configurable()
def load(data_dir, dataset_name):
    tf_record_dir = os.path.join(data_dir, "TfRecord")
    train_path_record = os.path.join(tf_record_dir, 'librispeech-train-00000-of-00001.parquet')
    val_path_record = os.path.join(tf_record_dir, 'librispeech-val-00000-of-00001.parquet')
    test_path_record = os.path.join(tf_record_dir, 'librispeech-test-00000-of-00001.parquet')
    if not os.path.exists(tf_record_dir):
        os.makedirs(tf_record_dir)
        train_text_path = glob('{}/train-clean-100/**/**/**/*.txt'.format(data_dir))
        val_text_path = glob('{}/dev-clean/**/**/**/*.txt'.format(data_dir))
        test_text_path = glob('{}/test-clean/**/**/**/*.txt'.format(data_dir))

        train_flac_path = glob('{}/train-clean-100/**/**/**/*.flac'.format(data_dir))
        val_flac_path = glob('{}/dev-clean/**/**/**/*.flac'.format(data_dir))
        test_flac_path = glob('{}/test-clean/**/**/**/*.flac'.format(data_dir))

        create_parquet_record(train_path_record, train_text_path, train_flac_path)
        create_parquet_record(val_path_record, val_text_path, val_flac_path)
        create_parquet_record(test_path_record, test_text_path, test_flac_path)

        # preprocess = PreProcessWav2Vec2(processor)
        # create_tf_records("train", train_samples, tf_record_dir, preprocess)
        # create_tf_records("val", val_samples, tf_record_dir, preprocess)
        # create_tf_records("test", test_samples, tf_record_dir, preprocess)

    # ds_train = load_tf_record(tf_record_dir, "train")
    # ds_val = load_tf_record(tf_record_dir, "val")
    # ds_test = load_tf_record(tf_record_dir, "test")
    dataset = load_dataset("parquet",
                           data_files={
                               'train': train_path_record, 'test': test_path_record, 'val': val_path_record})
    # ds_train = PytorchDataLoader(tf_train)
    # ds_val = PytorchDataLoader(tf_val)
    # ds_test = PytorchDataLoader(tf_test)
    return dataset


def create_parquet_record(path_record, text_path, flac_path):
    dataframe = pd.DataFrame(fetch_sound_text_mapping(text_path, flac_path))
    dataframe.to_parquet(path_record, index=False, compression='gzip')


def read_txt_file(f):
    with open(f, "r") as f:
        samples = f.read().split("\n")
        samples = {s.split()[0]: " ".join(s.split()[1:]) for s in samples if len(s.split()) > 2}
    return samples


def read_flac_file(file_path):
    with open(file_path, "rb") as f:
        info = sf.info(file_path)
        audio, sample_rate = sf.read(f)
    if sample_rate != REQUIRED_SAMPLE_RATE:
        raise ValueError(
            f"sample rate (={sample_rate}) of your files must be {REQUIRED_SAMPLE_RATE}"
        )
    if info.duration > 10:
        print(
            f"Too long, duration  (={info.duration})"
        )
    file_id = os.path.split(file_path)[-1][:-len(".flac")]
    return {file_id: audio}


def fetch_sound_text_mapping(txt_files, flac_files):
    txt_samples = {}
    for f in txt_files:
        txt_samples.update(read_txt_file(f))

    speech_samples = {}
    for f in flac_files:
        speech_samples.update(read_flac_file(f))

    assert len(txt_samples) == len(speech_samples)

    samples = [{
        'audio': speech_samples[file_id],
        'text': txt_samples[file_id],
        'audio_length': len(speech_samples[file_id])
    } for file_id in speech_samples.keys() if
        len(speech_samples[file_id]) < AUDIO_MAXLEN and len(txt_samples[file_id]) < LABEL_MAXLEN]
    return samples


# Create a Dataset from the DataFrame
class AudioDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.df.iloc[idx]['audio']
        transcript = self.df.iloc[idx]['text']
        audio_length = self.df.iloc[idx]['audio_length']
        return {'audio': audio_file, 'text': transcript, 'audio_length': audio_length}

#
# # def inputs_generator(samples):
# #     for speech, text in samples:
# #         yield preprocess.preprocess_speech(speech), preprocess.preprocess_text(text)
#
#
# def create_tf_records(file_type, samples, tf_record_dir, preprocess):
#     path_tf_record = os.path.join(tf_record_dir, 'librispeech-{}.tfrecord-00000-of-00001'
#                                   .format(file_type))
#     with tf.io.TFRecordWriter(path_tf_record) as writer:
#         for sample in samples:
#             audio_sample = tf.io.serialize_tensor(preprocess.preprocess_speech(sample[0]))
#             text_sample = tf.io.serialize_tensor(preprocess.preprocess_text(sample[1]))
#             writer.write(
#                 serialize_example(audio_sample.numpy(), text_sample.numpy()))
#
#
# def _bytes_feature(value):
#     """Returns a bytes_list from a string / byte."""
#     if isinstance(value, type(tf.constant(0))):
#         value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#
# def serialize_example(audio, text):
#     """
#   Creates a tf.train.Example message ready to be written to a file.
#   """
#     # Create a dictionary mapping the feature name to the tf.train.Example-compatible
#     # data type.
#     feature = {
#         'speech': _bytes_feature(audio),
#         'label': _bytes_feature(text),
#     }
#
#     # Create a Features message using tf.train.Example.
#     example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
#     return example_proto.SerializeToString()
#
#
# class PytorchDataLoader:
#     def __init__(self, tf_object, batch_size=BATCH_SIZE):
#         self.ds = tfds.as_numpy(tf_object)
#         self.num_examples = len(list(tf_object))
#         self.batch_size = batch_size
#         self.labeled = True
#         self._iterator = None
#
#     def __iter__(self):
#         if self._iterator is None:
#             self._iterator = iter(self.ds)
#         else:
#             self._reset()
#         return self._iterator
#
#     def _reset(self):
#         self._iterator = iter(self.ds)
#
#     def __next__(self):
#         batch = next(self._iterator)
#         return batch
#
#     def __len__(self):
#         return self.num_examples
#         # n_batches = self.num_examples // self.batch_size
#         # if self.num_examples % self.batch_size == 0:
#         #     return n_batches
#         # else:
#         #     return n_batches + 1
#
#
# # Define a function that returns the number of elements in a tensor
# def get_size(tensor):
#     return tf.size(tensor)
#
#
# def load_tf_record(tf_record_dir, file_type, batch_size=BATCH_SIZE):
#     dataset = tf.data.TFRecordDataset(
#         os.path.join(tf_record_dir, "librispeech-{}.tfrecord-00000-of-00001".format(file_type))).map(
#         read_tfrecords, num_parallel_calls=tf.data.AUTOTUNE)
#     # Use map() to apply the function to each tensor in the dataset
#     # ds = dataset.map(get_size)
#     # Use reduce() to find the maximum value
#     # max_size = ds.reduce(tf.keras.backend.max)
#
#     # print(max_size)  # Output: 4
#     # dataset = dataset.batch(batch_size)
#     # dataset = dataset.padded_batch(BATCH_SIZE, padded_shapes=(AUDIO_MAXLEN, LABEL_MAXLEN), padding_values=(0.0, 0))
#     return dataset.prefetch(tf.data.AUTOTUNE)
#
#
# # Read the data back out.
# def read_tfrecords(record: tf.train.Example):
#     desc = {
#         "speech": tf.io.FixedLenFeature((), tf.string),
#         "label": tf.io.FixedLenFeature((), tf.string),
#     }
#     record = tf.io.parse_single_example(record, desc)
#
#     speech = tf.io.parse_tensor(record["speech"], out_type=tf.float32)
#     label = tf.io.parse_tensor(record["label"], out_type=tf.int32)
#
#     return speech, label
#
#
# class LoadTfRecord(object):
#     def __init__(self, tf_record_dir, file_type, batch_size=BATCH_SIZE):
#         self.tf_record_dir = tf_record_dir
#         self.file_type = file_type
#         # self.count = 0
#         self.dataset = tf.data.TFRecordDataset(
#             os.path.join(self.tf_record_dir, "librispeech-{}.tfrecord-00000-of-00001".format(self.file_type))).map(
#             self.read_tfrecords, num_parallel_calls=tf.data.AUTOTUNE)
#         self.dataset = self.dataset.batch(batch_size)
#         self.dataset = self.dataset.padded_batch(
#             BATCH_SIZE, padded_shapes=(AUDIO_MAXLEN, LABEL_MAXLEN), padding_values=(0.0, 0))
#         self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)
#         self.count = len(list(self.dataset))
#
#     # def load_tf_record(self):
#     #     self.dataset = tf.data.TFRecordDataset(
#     #         os.path.join(self.tf_record_dir, "librispeech-{}.tfrecord-00000-of-00001".format(self.file_type))).map(
#     #         self.read_tfrecords, num_parallel_calls=tf.data.AUTOTUNE)
#     #     self.dataset = self.dataset.padded_batch(
#     #         BATCH_SIZE, padded_shapes=(AUDIO_MAXLEN, LABEL_MAXLEN), padding_values=(0.0, 0))
#     #     self.dataset = self.dataset.prefetch(tf.data.AUTOTUNE)
#
#     # Read the data back out.
#     def read_tfrecords(self, record: tf.train.Example):
#         # self.count = self.count + 1
#         desc = {
#             "speech": tf.io.FixedLenFeature((), tf.string),
#             "label": tf.io.FixedLenFeature((), tf.string),
#         }
#         record = tf.io.parse_single_example(record, desc)
#
#         speech = tf.io.parse_tensor(record["speech"], out_type=tf.float32)
#         label = tf.io.parse_tensor(record["label"], out_type=tf.int32)
#
#         return speech, label
