import gin
import logging
import tensorflow as tf
from absl import app, flags
from input_pipeline import datasets
from evaluation.eval import DisplayOutputs
from models.architectures import Transformer
from models.learning_rate_schedule import CustomSchedule
from utils import utils_params, utils_misc
from train import Trainer

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('runId', "", 'Specify path to the run directory.')


def main(argv):

    model_name = "transformer"

    # generate folder structures
    run_paths = utils_params.gen_run_folder(FLAGS.runId)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    train_data, val_data, test_data, data_dir = datasets.load()

    if model_name == "deep_speech":
        datasets.deepspeech_save_data(data_dir, train_data, val_data, test_data)
        return
    elif model_name == "transformer":
        train_ds, idx_to_char = datasets.preprocess_data(model_name, train_data, 32)
        val_ds, _ = datasets.preprocess_data(model_name, val_data, 6)
        test_ds, _ = datasets.preprocess_data(model_name, test_data, 6)

        batch = next(iter(val_ds))

        wandb_key = "8b5621f60202d49f7fa98ffafcb02ebbe4a3a314"

        display_cb = DisplayOutputs(
            batch, idx_to_char, target_start_token_idx=2, target_end_token_idx=3
        )  # set the arguments as per vocabulary index for '<' and '>'

        model = Transformer(
            num_hid=200,
            num_head=2,
            num_feed_forward=400,
            target_maxlen=200,
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
            steps_per_epoch=len(train_ds),
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        trainer = Trainer(model, optimizer, loss_fn, 1, train_ds, val_ds, display_cb, wandb_key)
        trainer.train()



if __name__ == "__main__":
    app.run(main)
