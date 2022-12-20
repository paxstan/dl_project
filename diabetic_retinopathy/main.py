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
flags.DEFINE_boolean('ensemble', False, 'Specify whether to train or evaluate a model.')
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

    # model for name, model in models:
    efficient_b4_model = efficient_netB4_model(input_shape=ds_info.features["image"].shape,
                                               n_classes=ds_info.features["label"].num_classes)

    res_net_model = res_net50_model(input_shape=ds_info.features["image"].shape,
                                    n_classes=ds_info.features["label"].num_classes)

    vgg_16_model = vgg16_model(input_shape=ds_info.features["image"].shape,
                               n_classes=ds_info.features["label"].num_classes)

    models = {
        efficient_b4_model.name: [1e5, efficient_b4_model, 'ensemble_1_top_conv'],
        res_net_model.name: [5e4, res_net_model, 'ensemble_2_conv5_block3_3_conv'],
        vgg_16_model.name: [1e5, vgg_16_model, 'ensemble_3_block5_conv2']
    }

    ensemble = Ensemble(run_paths)

    if FLAGS.train:
        train_routine(models, ds_train, ds_val, ds_info, run_paths, train=False)
        models_loaded = ensemble.load_all_models(models)
        if models_loaded:
            ensemble.define_stacked_model(n_classes=ds_info.features["label"].num_classes, dense_units=10)
            ensemble_models = {'ensemble': [1e4, ensemble.ensemble_model]}
            train_routine(ensemble_models, ds_train, ds_val, ds_info, run_paths)

    else:
        ensemble.all_models = [efficient_b4_model, res_net_model, vgg_16_model]
        ensemble.define_stacked_model(n_classes=ds_info.features["label"].num_classes, dense_units=10)
        if ensemble.load_all_models({'ensemble': [0, ensemble.ensemble_model, '']}):
            evaluation = Evaluation(ensemble.ensemble_model, ds_test, ds_info)
            evaluation.evaluate(FLAGS.ensemble)
            model_eval_routine(models, ensemble)


def train_routine(models, ds_train, ds_val, ds_info, run_paths, train=True):
    if train:
        for name, (steps, model, _) in models.items():
            trainer = Trainer(model, name, ds_train, ds_val, ds_info, run_paths, total_steps=steps)
            for _ in trainer.train():
                continue
            trainer.run.finish()


def model_eval_routine(models, ensemble):
    for name, (_, model, last_layer) in models.items():
        image_path = '/home/paxstan/Documents/Uni/DL Lab/idrid/IDRID_dataset/images/test/IDRiD_001.jpg'
        selected_model = [models for models in ensemble.all_models if models.name == model.name]
        grad_cam = GradCam(model=selected_model[0])
        img_array = grad_cam.get_img_array(image_path)
        predict = selected_model[0].predict(img_array)
        print('predicted class: ', np.argmax(predict))
        heatmap = grad_cam.make_gradcam_heatmap(img_array=img_array, last_conv_layer_name=last_layer)

        # Display heatmap
        plt.matshow(heatmap)
        plt.show()
        grad_cam.save_and_display_gradcam(img_path=image_path,
                                          heatmap=heatmap,
                                          cam_path=('/home/paxstan/Documents/Uni/DL_Lab/gradcam_result/IDRiD_001_{}.jpg'
                                                    ).format(name))


if __name__ == "__main__":
    app.run(main)
