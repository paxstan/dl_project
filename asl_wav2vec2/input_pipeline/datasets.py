import gin
import os
from glob import glob
# import soundfile as sf
import tensorflow as tf
from input_pipeline.preprocessing import PreProcess

REQUIRED_SAMPLE_RATE = 16000
AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256
BATCH_SIZE = 2
SEED = 42
preprocess = PreProcess()


@gin.configurable()
def load(data_dir, dataset_name):
    tf_record_dir = os.path.join(data_dir, "TfRecord")
    if not os.path.exists(tf_record_dir):
        os.makedirs(tf_record_dir)
        train_text_path = glob('{}/train-clean-100/**/**/**/*.txt'.format(data_dir))
        val_text_path = glob('{}/dev-clean/**/**/**/*.txt'.format(data_dir))
        test_text_path = glob('{}/test-clean/**/**/**/*.txt'.format(data_dir))

        train_flac_path = glob('{}/train-clean-100/**/**/**/*.flac'.format(data_dir))
        val_flac_path = glob('{}/dev-clean/**/**/**/*.flac'.format(data_dir))
        test_flac_path = glob('{}/test-clean/**/**/**/*.flac'.format(data_dir))

        train_samples = fetch_sound_text_mapping(train_text_path, train_flac_path)
        val_samples = fetch_sound_text_mapping(val_text_path, val_flac_path)
        test_samples = fetch_sound_text_mapping(test_text_path, test_flac_path)

        create_tf_records("train", train_samples, tf_record_dir)
        create_tf_records("val", test_samples, tf_record_dir)
        create_tf_records("test", val_samples, tf_record_dir)

    ds_train = load_tf_record(tf_record_dir, "train")
    ds_val = load_tf_record(tf_record_dir, "val")
    ds_test = load_tf_record(tf_record_dir, "test")
    return ds_train, ds_val, ds_test


def load_tf_record(tf_record_dir, file_type):
    dataset = tf.data.TFRecordDataset(
        os.path.join(tf_record_dir, "librispeech-{}.tfrecord-00000-of-00001".format(file_type))).map(
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


def read_txt_file(f):
    with open(f, "r") as f:
        samples = f.read().split("\n")
        samples = {s.split()[0]: " ".join(s.split()[1:]) for s in samples if len(s.split()) > 2}
    return samples


def read_flac_file(file_path):
    # with open(file_path, "rb") as f:
    #     audio, sample_rate = sf.read(f)
    # if sample_rate != REQUIRED_SAMPLE_RATE:
    #     raise ValueError(
    #         f"sample rate (={sample_rate}) of your files must be {REQUIRED_SAMPLE_RATE}"
    #     )
    # file_id = os.path.split(file_path)[-1][:-len(".flac")]
    # return {file_id: audio}


def fetch_sound_text_mapping(txt_files, flac_files):
    txt_samples = {}
    for f in txt_files:
        txt_samples.update(read_txt_file(f))

    speech_samples = {}
    for f in flac_files:
        speech_samples.update(read_flac_file(f))

    assert len(txt_samples) == len(speech_samples)

    samples = [(speech_samples[file_id], txt_samples[file_id]) for file_id in speech_samples.keys() if
               len(speech_samples[file_id]) < AUDIO_MAXLEN]
    return samples


def inputs_generator(samples):
    for speech, text in samples:
        yield preprocess.preprocess_speech(speech), preprocess.preprocess_text(text)


def create_tf_records(file_type, samples, tf_record_dir):
    path_tf_record = os.path.join(tf_record_dir, 'librispeech-{}.tfrecord-00000-of-00001'
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
