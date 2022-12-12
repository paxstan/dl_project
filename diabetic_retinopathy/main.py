import gin
import logging
from absl import app, flags
import wandb

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like, dense_net121_model, res_net101_model, xception_model, res_net50_model, nas_net
from models.architectures import efficient_netB4_model, inceptionv3_model
FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # initialize wandb with your project name and optionally with configutations.
    # play around with the config values and see the result on your wandb dashboard.

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # model
    # model = vgg_like(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
    #model = dense_net121_model(input_shape=ds_info.features["image"].shape,n_classes=ds_info.features["label"].num_classes)
    #model = res_net101_model(input_shape=ds_info.features["image"].shape,
      #                       n_classes=ds_info.features["label"].num_classes)
    # model = xception_model(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
    # model = res_net50_model(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
    # model = nas_net(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)
    #model = efficient_netB4_model(input_shape=ds_info.features["image"].shape,n_classes=ds_info.features["label"].num_classes)
    model = inceptionv3_model(input_shape=ds_info.features["image"].shape,
                                  n_classes=ds_info.features["label"].num_classes)

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        config = {
            "learning_rate": trainer.learning_rate,
            "epochs": trainer.total_steps,
            "batch_size": trainer.batch_size,
            "log_step": trainer.log_interval,
            "val_log_step": 0,
            "architecture": "CNN",
            "dataset": "IDRID"
        }
        wandb.login(anonymous="allow", key=trainer.wandb_key)
        run = wandb.init(project='idrid-test', config=config)
        config = wandb.config
        for log in trainer.train():
            wandb.log(log)
            continue
        run.finish()
    else:
        evaluate(model,
                 checkpoint,
                 ds_test,
                 ds_info,
                 run_paths)


if __name__ == "__main__":
    app.run(main)
