import gin
import logging

import numpy as np
from absl import app, flags

from train import Trainer
from evaluation.eval import Evaluation
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import res_net50_model, efficient_netB4_model, vgg16_model
from models.ensemble import Ensemble
from visualization.gradcam import GradCam
import matplotlib.pyplot as plt

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
    ds_train, ds_val, ds_test, ds_info = datasets.load()

    # the three chosen models
    efficient_b4_model = efficient_netB4_model(input_shape=ds_info.features["image"].shape,
                                               n_classes=ds_info.features["label"].num_classes)

    res_net_model = res_net50_model(input_shape=ds_info.features["image"].shape,
                                    n_classes=ds_info.features["label"].num_classes)

    vgg_16_model = vgg16_model(input_shape=ds_info.features["image"].shape,
                               n_classes=ds_info.features["label"].num_classes)

    # A models dictionary with each model having their steps for training
    # and the layer for extract grad cam result mentioned
    models = {
        efficient_b4_model.name: [1e5, efficient_b4_model, 'top_conv'],
        res_net_model.name: [5e4, res_net_model, 'conv5_block3_3_conv'],
        vgg_16_model.name: [1e5, vgg_16_model, 'block5_conv2']
    }

    if FLAGS.train:
        # to train each models individual
        train_routine(models, ds_train, ds_val, ds_info, run_paths)

    else:
        # evaluate ensemble of model based on voting method
        ensemble = Ensemble(run_paths)
        models_loaded = ensemble.load_all_models(models)
        if models_loaded:
            evaluation = Evaluation(ensemble, ds_test, ds_info, "ensemble")
            evaluation.evaluate(ensemble=True)

            # evaluate each model individually
            model_eval_routine(models, ensemble, ds_test, ds_info)


def train_routine(models, ds_train, ds_val, ds_info, run_paths):
    """To run train routine for list of model in iteration"""
    for name, (steps, model, _) in models.items():
        trainer = Trainer(model, name, ds_train, ds_val, ds_info, run_paths, total_steps=steps)
        for _ in trainer.train():
            continue
        trainer.run.finish()


@gin.configurable
def model_eval_routine(models, ensemble, ds_test, ds_info, image_path, grad_path):
    """To evaluate individual model in iteration to get accuracy and grad cam result"""
    for name, (_, model, last_layer) in models.items():
        # get each model
        selected_model = [models for models in ensemble.all_models if models.name == model.name]
        logging.info(f"Model: {model.name}")
        evaluation = Evaluation(selected_model[0], ds_test, ds_info, name)
        evaluation.evaluate()
        # perform grad cam on the selected model
        grad_cam = GradCam(model=selected_model[0])
        img_array = grad_cam.get_img_array(image_path)
        predict = selected_model[0].predict(img_array)
        logging.info(f"predicted class: {np.argmax(predict).tostring()}")
        heatmap = grad_cam.make_gradcam_heatmap(img_array=img_array, last_conv_layer_name=last_layer)
        # Display heatmap
        plt.matshow(heatmap)
        plt.show()
        grad_cam_path = '{}/IDRiD_001_{}.jpg'.format(grad_path, name)
        logging.info(f"Saving grad cam image at {grad_cam_path}")
        # save grad cam result
        grad_cam.save_and_display_gradcam(img_path=image_path,
                                          heatmap=heatmap,
                                          cam_path=grad_cam_path)


if __name__ == "__main__":
    app.run(main)
