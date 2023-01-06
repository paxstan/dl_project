import gin
import os
import logging
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds

from input_pipeline.preprocessing import preprocess, augment


@gin.configurable
def load(name, data_dir, tf_record_dir):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        if not os.path.exists(tf_record_dir):
            os.makedirs(tf_record_dir)
            train_size = create_tf_records("train", data_dir, tf_record_dir)
            test_size = create_tf_records("test", data_dir, tf_record_dir)
            tfrecord_to_tfds(tf_record_dir, train_size, test_size)
        tfds_builder = tfds.builder_from_directory(tf_record_dir)
        ds_info = tfds_builder.info
        ds_train, ds_val, ds_test = tfds_builder.as_dataset(
            split=['train[:80%]', 'train[20%:]', 'test'],
            shuffle_files=True,
            as_supervised=True
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        def _preprocess(img_label_dict):
            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    # Prepare training dataset
    if caching:
        ds_train = ds_train.cache()
    ds_train = ds_train.map(
        (lambda x, y: augment(x, y)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.map(
        (lambda x, y: (tf.image.resize(x, size=(256, 256), preserve_aspect_ratio=True), y)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.map(
        (lambda x, y: (tf.image.resize(x, size=(256, 256), preserve_aspect_ratio=True), y)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info


def rawdata_to_df(data_dir, file_type, fields):
    path_csv = os.path.join(data_dir, "labels", '{}.csv'.format(file_type))
    data_frame = pd.read_csv(path_csv, usecols=fields)
    data_frame['path'] = data_frame['Image name'].map(
        lambda x: os.path.join(data_dir, "images/{}".format(file_type), '{}.jpg'.format(x)))
    data_frame['exists'] = data_frame['path'].map(os.path.exists)
    data_frame.dropna(inplace=True)
    data_frame = data_frame[data_frame['exists']]
    return data_frame


def balance_data(class_size, df):
    new_df = df.groupby(['Retinopathy grade']).apply(lambda x: x.sample(class_size, replace=True)).reset_index(
        drop=True)
    new_df = new_df.sample(frac=1).reset_index(drop=True)
    print('New Data Size:', new_df.shape[0], 'Old Size:', df.shape[0])
    new_df['Retinopathy grade'].hist(figsize=(10, 5))
    return new_df


def create_tf_records(file_type, data_dir, tf_record_dir):
    fields = ['Image name', 'Retinopathy grade']
    path_tf_record = os.path.join(tf_record_dir, 'idrid-{}.tfrecord-00000-of-00001'
                                  .format(file_type))
    dataframe = rawdata_to_df(data_dir, file_type, fields)
    dataframe = balance_data(dataframe.pivot_table(index='Retinopathy grade', aggfunc=len).max().max(), dataframe)
    with tf.io.TFRecordWriter(path_tf_record) as writer:
        for index, rows in dataframe.iterrows():
            print(rows['Image name'])
            # label = 0 if (int(rows['Retinopathy grade']) < 2) else 1
            label = int(rows['Retinopathy grade'])
            image = preprocess(cv2.imread(rows['path']))
            png_img = cv2.imencode('.png', image)[1]
            np_final_image = np.array(png_img)
            writer.write(serialize_example(
                np_final_image.tobytes(),
                label))
    return dataframe.shape[0]


# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image, label):
    """
  Creates a tf.train.Example message ready to be written to a file.
  """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def tfrecord_to_tfds(path, train_size, test_size):
    features = tfds.features.FeaturesDict({
        'image':
            tfds.features.Image(shape=(256, 256, 3)),
        'label':
            tfds.features.ClassLabel(names=['0', '1', '2', '3', '4']),
    })

    split_infos = [
        tfds.core.SplitInfo(
            name='train',
            shard_lengths=[train_size],  # Num of examples in shard0, shard1,...
            num_bytes=0,  # Total size of your dataset (if unknown, set to 0)
        ),
        tfds.core.SplitInfo(
            name='test',
            shard_lengths=[test_size],  # Num of examples in shard0, shard1,...
            num_bytes=0,  # Total size of your dataset (if unknown, set to 0)
        ),
    ]

    tfds.folder_dataset.write_metadata(
        data_dir=path,
        features=features,
        split_infos=split_infos,
        description="""IDRID Dataset""",
        supervised_keys=('image', 'label')

    )
