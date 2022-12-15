import gin
import logging

import numpy as np
from absl import app, flags
import argparse

from train import Trainer
from evaluation.eval import Evaluation
# from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import efficient_netB4_model, res_net50_model, vgg16_model
from visualization.gradcam import GradCam
import matplotlib.pyplot as plt

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
    efficient_b4_model = efficient_netB4_model(input_shape=ds_info.features["image"].shape,
                                               n_classes=ds_info.features["label"].num_classes)

    res_net_model = res_net50_model(input_shape=ds_info.features["image"].shape,
                                    n_classes=ds_info.features["label"].num_classes)

    vgg_16_model = vgg16_model(input_shape=ds_info.features["image"].shape,
                               n_classes=ds_info.features["label"].num_classes)

    models = {
        'efficient_net_b4': efficient_b4_model,
        'res_net_50': res_net_model,
        'vgg_16': vgg_16_model
    }

    if FLAGS.train:
        for name, model in models:
            trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
            trainer.train_and_checkpoint()
    else:
        evaluation = Evaluation(model, ds_test, ds_info, run_paths)
        evaluation.check_loaded_weights()
        evaluation.evaluate()
        image_path = '/home/paxstan/Documents/Uni/DL Lab/idrid/IDRID_dataset/images/test/IDRiD_001.jpg'
        grad_cam = GradCam(model=evaluation.model)
        # trained_model = evaluation.model.get_layer('inception_resnet_v2')
        # grad_cam = GradCam(model=trained_model)
        img_array = grad_cam.get_img_array(image_path)
        predict = evaluation.model.predict(img_array)
        print('predicted class: ', np.argmax(predict))
        heatmap = grad_cam.make_gradcam_heatmap(img_array=img_array, last_conv_layer_name='top_conv')

        # Display heatmap
        plt.matshow(heatmap)
        plt.show()
        grad_cam.save_and_display_gradcam(img_path=image_path,
                                          heatmap=heatmap,
                                          cam_path='/home/paxstan/Documents/Uni/DL Lab/idrid/IDRiD_001_cam_ef.jpg')


if __name__ == "__main__":
    app.run(main)
