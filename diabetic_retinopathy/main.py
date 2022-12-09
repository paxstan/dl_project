import gin
import logging
from absl import app, flags
import argparse

from train import Trainer
from evaluation.eval import Evaluation
# from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('runId', "", 'Specify whether to train or evaluate a model.')


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder(FLAGS.runId)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # initialize wandb with your project name and optionally with configurations.
    # play around with the config values and see the result on your wandb dashboard.

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # model
    model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        trainer.train_and_checkpoint()
    else:
        evaluation = Evaluation(model, ds_test, ds_info, run_paths)
        evaluation.evaluate()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--train", dest="train", type=str2bool, required=True)
    # parser.add_argument("--runId", dest="run_id", type=str)
    # args = parser.parse_args()
    # flags.DEFINE_boolean('train', args.train, 'Specify whether to train or evaluate a model.')
    # flags.DEFINE_string('run_id', args.run_id, 'Specify the run id')
    app.run(main)
