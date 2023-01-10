import gin
import logging
import tensorflow as tf
from absl import app, flags
from input_pipeline import datasets
from evaluation.eval import DisplayOutputs
from models.architectures import Transformer
from models.learning_rate_schedule import CustomSchedule
from utils import utils_params, utils_misc

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('runId', "", 'Specify path to the run directory.')

def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder(FLAGS.runId)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds, val_ds, vectorizer, max_target_len = datasets.load()

    batch = next(iter(val_ds))

    # The vocabulary to convert predicted indices into characters
    idx_to_char = vectorizer.get_vocabulary()
    display_cb = DisplayOutputs(
        batch, idx_to_char, target_start_token_idx=2, target_end_token_idx=3
    )  # set the arguments as per vocabulary index for '<' and '>'

    model = Transformer(
        num_hid=200,
        num_head=2,
        num_feed_forward=400,
        target_maxlen=max_target_len,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=34,
    )
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=0.1,
    )

    learning_rate = CustomSchedule(
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=len(ds),
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss_fn)

    history = model.fit(ds, validation_data=val_ds, callbacks=[display_cb], epochs=1)


if __name__ == "__main__":
    app.run(main)
