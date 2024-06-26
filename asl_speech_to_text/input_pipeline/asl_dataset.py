import gin
import os
from glob import glob
import soundfile as sf
import pandas as pd
from datasets import load_dataset
import itertools

REQUIRED_SAMPLE_RATE = 16000
AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256


@gin.configurable()
class LoadDataset(object):
    def __init__(self, data_dir, processor):
        self.data_dir = data_dir
        self.processor = processor
        self.tf_record_dir = os.path.join(data_dir, "TfRecord")
        self.train_path_record = os.path.join(self.tf_record_dir, 'train')
        self.val_path_record = os.path.join(self.tf_record_dir, 'val')
        self.test_path_record = os.path.join(self.tf_record_dir, 'test')

    def load(self):
        """
        Function to load dataset from the parquet files
        Returns:
            train, validation and test dataset
        """
        if not os.path.exists(self.tf_record_dir):
            os.makedirs(self.train_path_record)
            os.makedirs(self.val_path_record)
            os.makedirs(self.test_path_record)
            train_data_dir = get_dirs_from_path(os.path.join(self.data_dir, "train-clean-100/train-clean-100"), 20)
            val_data_dir = get_dirs_from_path(os.path.join(self.data_dir, "dev-clean/dev-clean"), 5)
            test_data_dir = get_dirs_from_path(os.path.join(self.data_dir, "test-clean/test-clean"), 5)

            self.create_parquet_record(self.train_path_record, "train", train_data_dir)
            self.create_parquet_record(self.val_path_record, "val", val_data_dir)
            self.create_parquet_record(self.test_path_record, "test", test_data_dir)

        dataset = load_dataset("parquet",
                               data_files={
                                   'train': glob('{}/*.parquet'.format(self.train_path_record)),
                                   'test': glob('{}/*.parquet'.format(self.test_path_record)),
                                   'val': glob('{}/*.parquet'.format(self.val_path_record))})
        ds_train = dataset["train"]
        ds_val = dataset["val"]
        ds_test = dataset["test"]
        return ds_train, ds_val, ds_test

    def prepare_dataset(self, batch):
        """
        Function to preprocess the dataset
        Args:
            batch: batch of dataset to preprocess using processor

        Returns:
            preprocessed batch
        """
        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = self.processor(batch["audio"], sampling_rate=REQUIRED_SAMPLE_RATE).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["text"]).input_ids
        return batch

    def create_parquet_record(self, record_dir, file_type, data_list):
        """
        Function to create parquet files to store the dataset
        Args:
            record_dir:
            file_type:
            data_list:

        Returns:

        """
        for key, value in data_list:
            text_paths = []
            flac_paths = []
            for dir_path in value:
                text_paths.append(glob('{}/**/*.txt'.format(dir_path[1])))
                flac_paths.append(glob('{}/**/*.flac'.format(dir_path[1])))
            dataframe = pd.DataFrame(self.fetch_sound_text_mapping(
                list(itertools.chain(*text_paths)), list(itertools.chain(*flac_paths))))
            dataframe.to_parquet(
                "{}/librispeech-{}-00000-of-0000{}.parquet".format(record_dir, file_type, key + 1),
                index=False, compression='gzip')

    def fetch_sound_text_mapping(self, txt_files, flac_files):
        """
        Function to map the transcript and its corresponding audio
        Args:
            txt_files: list of transcription files
            flac_files: list of FLAC format audio files

        Returns:
            mapped set of transcription and audio
        """
        txt_samples = {}
        for f in txt_files:
            txt_samples.update(read_txt_file(f))

        speech_samples = {}
        for f in flac_files:
            speech_samples.update(read_flac_file(f))

        txt_samples, speech_samples = remove_mismatch(txt_samples, speech_samples)

        samples = [{
            'input_values': self.processor(speech_samples[file_id], sampling_rate=REQUIRED_SAMPLE_RATE).input_values[0],
            'input_length': len(speech_samples[file_id]),
            'labels': self.processor(text=txt_samples[file_id]).input_ids
        } for file_id in speech_samples.keys() if
            len(speech_samples[file_id]) < AUDIO_MAXLEN and len(txt_samples[file_id]) < LABEL_MAXLEN]
        return samples


def get_dirs_from_path(path, count):
    """
    Function to get directories from the path
    Args:
        path: path to the root directory
        count: limit the number of directories to be considered

    Returns:
        list of set of directories
    """
    dirs = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if len(dirs) > count:
        dirs = dirs[:count]
    return [(k, list(g)) for k, g in itertools.groupby(enumerate(dirs), lambda x: x[0] // 2)]


def read_txt_file(f):
    """
    Function to read from file
    Args:
        f: path to the file to be read from

    Returns:
        set of mapped transcript and its audio name
    """
    with open(f, "r") as f:
        samples = f.read().split("\n")
        samples = {s.split()[0]: " ".join(s.split()[1:]) for s in samples if len(s.split()) > 5}
    return samples


def read_flac_file(file_path):
    """
    To read the audio array from FLAC audio file
    Args:
        file_path: path to the audio file

    Returns:
        set of audio array with its name
    """
    with open(file_path, "rb") as f:
        audio, sample_rate = sf.read(f)
    if sample_rate != REQUIRED_SAMPLE_RATE:
        raise ValueError(
            f"sample rate (={sample_rate}) of your files must be {REQUIRED_SAMPLE_RATE}"
        )
    file_id = os.path.split(file_path)[-1][:-len(".flac")]
    return {file_id: audio}


def remove_mismatch(dict1, dict2):
    """
    To remove mismatched elements from two sets based on their keys
    Args:
        dict1: txt_sample dict
        dict2: speech_sample dict

    Returns:
        txt_sample and speech_sample with their mismatched key and element removed
    """
    common_keys = set(dict1.keys()) & set(dict2.keys())
    dict1 = {k: v for k, v in dict1.items() if k in common_keys}
    dict2 = {k: v for k, v in dict2.items() if k in common_keys}
    return dict1, dict2
