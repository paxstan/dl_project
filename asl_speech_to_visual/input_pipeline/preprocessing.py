import re

import gin
import numpy as np
from wav2vec2 import Wav2Vec2Processor
import tensorflow as tf
import os
import string
import csv
import argparse
# import librosa  # pip install librosa==0.7.2
# import num2words  # pip install num2words


class VectorizeChar:
    def __init__(self, max_len=200):
        # self.vocab = (
        #         ["-", "#", "<", ">"]
        #         + [chr(i + 96) for i in range(1, 27)]
        #         + [" ", ".", ",", "?"]
        # )
        self.vocab = (
                [chr(i + 96) for i in range(1, 27)]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        ftext = replace_func(text).replace("  ", " ").strip()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab


def preprocess(data, sample_rate):
    try:
        # Separate file name and transcript from metadata file. Preprocess transcript and get audio info too
        # convert all numbers to text using num2words

        fname = data['wav_filename']
        ftext = data['transcript'].strip().lower()
        ftext = replace_func(ftext).replace("  ", " ").strip()
        # ftext = re.sub(r"(\d+)", lambda x: num2words.num2words(int(x.group(0))), ftext)
        fdur, fsr = get_audio_info(fname)

        # Don't add files which don't fit into model specifications
        # Either not 16kHz or longer than 10 secs or empty txt
        if fsr != 16000 or fdur > 10 or ftext == '':
            return np.nan
        else:
            data['transcript'] = ftext
            return data
    except Exception as e:
        print(str(e))


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

    # return librosa.get_duration(filename=file_name), librosa.get_samplerate(file_name)
