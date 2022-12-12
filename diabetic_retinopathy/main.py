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
from models.architectures import vgg_like, efficient_netB4_model,inception_resnetv2_model
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
    model = inception_resnetv2_model(input_shape=ds_info.features["image"].shape,
                                  n_classes=ds_info.features["label"].num_classes)

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        trainer.train_and_checkpoint()
    else:
        evaluation = Evaluation(model, ds_test, ds_info, run_paths)
        evaluation.evaluate()
        image_path = '/home/data/IDRID_dataset/images/test/IDRiD_001.jpg'
        grad_cam = GradCam(model=evaluation.model)
        # trained_model = evaluation.model.get_layer('inception_resnet_v2')
        # grad_cam = GradCam(model=trained_model)
        img_array = grad_cam.get_img_array(image_path)
        predict = evaluation.model.predict(img_array)
        print('predicted class: ', np.argmax(predict))
        heatmap = grad_cam.make_gradcam_heatmap(img_array=img_array, last_conv_layer_name='inception_resnet_v2')

        # Display heatmap
        plt.matshow(heatmap)
        plt.show()
        grad_cam.save_and_display_gradcam(img_path=image_path,
                                          heatmap=heatmap,
                                          cam_path='IDRiD_001_cam.jpg')


if __name__ == "__main__":
    app.run(main)
