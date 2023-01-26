import gin
import numpy as np
from wav2vec2 import Wav2Vec2Processor
import tensorflow as tf
import os
import string
import csv
import argparse
import librosa  # pip install librosa==0.7.2
import num2words  # pip install num2words
import re
import soundfile as sf
import tensorflow_io as tfio
# from transformers import Wav2Vec2Processor
import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'
REQUIRED_SAMPLE_RATE = 16000
AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256
BATCH_SIZE = 2


def remove_special_characters(text):
    text = re.sub(chars_to_ignore_regex, '', text).lower() + " "
    return text


def extract_all_chars(text):
    all_text = " ".join(text)
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


# def read_flac_file(file_path):
#     with open(file_path, "rb") as f:
#         audio, sample_rate = sf.read(f)
#     if sample_rate != REQUIRED_SAMPLE_RATE:
#         return np.nan
#         # raise ValueError(
#         #     f"sample rate (={sample_rate}) of your files must be {REQUIRED_SAMPLE_RATE}"
#         # )
#     else:
#         # audio = tf.squeeze(audio, axis=[-1])
#         stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
#         x = tf.math.pow(tf.abs(stfts), 0.5)
#         # normalisation
#         means = tf.math.reduce_mean(x, 1, keepdims=True)
#         stddevs = tf.math.reduce_std(x, 1, keepdims=True)
#         x = (x - means) / stddevs
#         audio_len = tf.shape(x)[0]
#         # padding to 10 seconds
#         pad_len = 2754
#         paddings = tf.constant([[0, pad_len], [0, 0]])
#         x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
#         file_id = os.path.split(file_path)[-1][:-len(".flac")]
#         return x


@tf.function
def path_to_audio(path):
    # spectrogram using stft
    audio = tf.io.read_file(path)
    # audio, sample_rate = sf.read(path)
    print(audio)
    # audio = tfio.audio.AudioIOTensor(path, dtype=tf.int16)
    audio, sp = tfio.audio.decode_flac(audio, dtype=tf.int16)
    # tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=[-1])
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


class VectorizeChar:
    def __init__(self):
        self.vocab = (
                ["<", ">"]
                + [chr(i + 96) for i in range(1, 27)]
                + ["'", " "]
        )
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = replace_func(text).replace("  ", " ").strip()
        if len(text) > LABEL_MAXLEN:
            return np.nan
        else:
            # text = text[: LABEL_MAXLEN - 2]
            text = "<" + text + ">"
            pad_len = LABEL_MAXLEN - len(text)
            return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab


def preprocess(data):
    try:
        # Separate file name and transcript from metadata file. Preprocess transcript and get audio info too
        # convert all numbers to text using num2words

        fname = data['wav_filename']
        ftext = data['transcript'].strip().lower()
        ftext = replace_func(ftext).replace("  ", " ").strip()
        ftext = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), ftext)
        fdur, fsr = get_audio_info(fname)

        # Don't add files which don't fit into model specifications
        # Either not 16kHz or longer than 10 secs or empty txt
        if fsr != 16000 or fdur > 10 or ftext == '':
            data['transcript'] = np.nan
        else:
            data['transcript'] = ftext
    except Exception as e:
        print(str(e))

    return data


def replace_func(text):
    """Remove extra characters from the transcript which are not in DeepSpeech's alphabet.txt
    """

    for ch in ['\\', '`', '‘', '’', '*', '_', ',', '"', '{', '}', '[', ']', '(', ')', '>', '#', '+', '-', '.', '!', '$',
               ':', ';', '|', '~', '@', '*', '<', '?', '/']:
        if ch in text:
            text = text.replace(ch, "")
        elif ch == '&':
            text = text.replace(ch, "and")

    return text


def get_audio_info(file_name):
    """Return specified audio file duration and sample rate
    """
    return librosa.get_duration(filename=file_name), librosa.get_samplerate(file_name)

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

class PreProcess(object):
    def __init__(self):
        self.tokenizer = Wav2Vec2Processor(is_tokenizer=True)
        self.processor = Wav2Vec2Processor(is_tokenizer=False)

    def preprocess_text(self, text):
        label = self.tokenizer(text)
        return tf.constant(label, dtype=tf.int32)

    def preprocess_speech(self, audio):
        audio = tf.constant(audio, dtype=tf.float32)
        return self.processor(tf.transpose(audio))

def fetch_sound_text_mapping(dataframe):
    txt_samples = {}
    speech_samples = {}
    for index, rows in dataframe.iterrows():
        txt_samples.update(
            {rows['wav_name']: rows['transcript']})
        speech_samples.update(read_flac_file(rows['wav_filename']))

    assert len(txt_samples) == len(speech_samples)

    samples = [(speech_samples[file_id], txt_samples[file_id]) for file_id in speech_samples.keys() if
               len(speech_samples[file_id]) < AUDIO_MAXLEN]
    return samples


def read_flac_file(file_path):
    with open(file_path, "rb") as f:
        audio, sample_rate = sf.read(f)
    if sample_rate != REQUIRED_SAMPLE_RATE:
        raise ValueError(
            f"sample rate (={sample_rate}) of your files must be {REQUIRED_SAMPLE_RATE}"
        )
    file_id = os.path.split(file_path)[-1][:-len(".flac")]
    return {file_id: audio}


def deepspeech_preprocess(data_frame):
    data_frame = data_frame.apply(preprocess, axis=1)
    data_frame.dropna(inplace=True)
    data_frame.drop(['wav_filename'], axis=1, inplace=True)
    data_frame.rename(columns={'wav_name': 'wav_filename'}, inplace=True)
    return data_frame
