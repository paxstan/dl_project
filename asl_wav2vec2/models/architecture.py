import tensorflow as tf
import tensorflow_hub as hub
from wav2vec2 import Wav2Vec2Config
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256
BATCH_SIZE = 2


class wav2vec2_tf(object):
    def __init__(self):
        self.config = Wav2Vec2Config()

        pretrained_layer = hub.KerasLayer("https://tfhub.dev/vasudevgupta7/wav2vec2/1", trainable=True)
        inputs = tf.keras.Input(shape=(AUDIO_MAXLEN,))
        hidden_states = pretrained_layer(inputs)
        outputs = tf.keras.layers.Dense(self.config.vocab_size)(hidden_states)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model(tf.random.uniform(shape=(BATCH_SIZE, AUDIO_MAXLEN)))
        self.model.summary()


class Wav2Vec2100h(object):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-100h")
