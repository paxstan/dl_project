import logging
import gin

import ray
from ray import tune

from input_pipeline.datasets import load
from models.architectures import efficient_netB4_model, res_net50_model
from train import Trainer
from utils import utils_params, utils_misc


def train_func(config):
    # Hyperparameters
    bindings = []
    for key, value in config.items():
        bindings.append(f'{key}={value}')

    # generate folder structures
    run_paths = utils_params.gen_run_folder(','.join(bindings))

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['/home/janavi/dl_lab/dl-lab-22w-team07/diabetic_retinopathy/configs/config.gin'],
                                        bindings)  # change path to absolute path of config file
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = load()

    # model
    model = res_net50_model(input_shape=ds_info.features["image"].shape, n_classes=ds_info.features["label"].num_classes)

    trainer = Trainer(model, "resnet50", ds_train, ds_val, ds_info, run_paths, total_steps=1e5)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


ray.init(num_cpus=10, num_gpus=1)
analysis = tune.run(
    train_func, num_samples=20, resources_per_trial={"cpu": 10, "gpu": 1},
    config={
        # "Trainer.total_steps": tune.grid_search([1e5]),
        "Trainer.learning_rate": tune.choice([0.0001, 0.001, 0.01, 0.1]),
        #"efficient_netB4_model.base_filters": tune.choice([8, 16]),
        # "vgg_like.n_blocks": tune.choice([2, 3, 4, 5]),
        "res_net50_model.dense_units": tune.choice([32, 64, 128, 256, 512]),
        "res_net50_model.dropout_rate": tune.uniform(0, 0.9),
    })

print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
