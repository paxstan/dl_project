import tensorflow as tf
import wav2vec2
import transformers
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


class PreProcess(object):
    def __init__(self):
        self.tokenizer = wav2vec2.Wav2Vec2Processor(is_tokenizer=True)
        self.processor = wav2vec2.Wav2Vec2Processor(is_tokenizer=False)

    def preprocess_text(self, text):
        label = self.tokenizer(text)
        return tf.constant(label, dtype=tf.int32)

    def preprocess_speech(self, audio):
        audio = tf.constant(audio, dtype=tf.float32)
        return self.processor(tf.transpose(audio))


class PreProcessWav2Vec2(object):
    def __init__(self, processor):
        self.processor = processor

    def preprocess_text(self, text):
        with self.processor.as_target_processor():
            labels = self.processor(text).input_ids
        return tf.constant(labels, dtype=tf.int32)

    def preprocess_speech(self, audio):
        audio = tf.constant(audio, dtype=tf.float32)
        return self.processor(tf.transpose(audio), sampling_rate=16000, return_tensors="pt").input_values[0]


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

    processor: transformers.Wav2Vec2Processor
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
