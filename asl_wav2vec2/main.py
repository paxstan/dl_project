import gin
import logging
import tensorflow as tf
from absl import app, flags
from utils import utils_params, utils_misc
from wav2vec2 import CTCLoss
from models.architecture import wav2vec2_tf
from input_pipeline.datasets import load

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('runId', "", 'Specify path to the run directory.')
LEARNING_RATE = 5e-5
AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256
BATCH_SIZE = 2

def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder(FLAGS.runId)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    ds_train, ds_val, ds_test = load()
    wav2vec2 = wav2vec2_tf()
    loss_fn = CTCLoss(wav2vec2.config, (BATCH_SIZE, AUDIO_MAXLEN), division_factor=BATCH_SIZE)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    wav2vec2.model.compile(optimizer, loss=loss_fn)
    history = wav2vec2.model.fit(ds_train, validation_data=ds_val, epochs=3)
    print(history.history)



if __name__ == "__main__":
    app.run(main)
