import gin
import os
from glob import glob
import soundfile as sf
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader

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
        dataset = dataset.map(self.prepare_dataset, num_proc=4)
        ds_train = dataset["train"].map(group_batch, batched=True, batch_size=32)
        ds_val = dataset["val"].map(group_batch, batched=True, batch_size=32)
        ds_test = dataset["test"].map(group_batch, batched=True, batch_size=32)
        return ds_train, ds_val, ds_test

    def prepare_dataset(self, batch):
        # batched output is "un-batched" to ensure mapping is correct
        batch["input_values"] = self.processor(batch["audio"], sampling_rate=REQUIRED_SAMPLE_RATE).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with self.processor.as_target_processor():
            batch["labels"] = self.processor(batch["text"]).input_ids
        return batch


def group_batch(batch):
    return {k: [v] for k, v in batch.items()}


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
