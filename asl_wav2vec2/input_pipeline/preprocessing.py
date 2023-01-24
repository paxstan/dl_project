import tensorflow as tf
from wav2vec2 import Wav2Vec2Processor


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
