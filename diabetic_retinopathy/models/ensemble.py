import gin
import tensorflow as tf
import logging
import os
import numpy as np


@gin.configurable
class Ensemble(object):
    """Class for ensemble voting"""
    def __init__(self, run_paths):
        self.run_paths = run_paths
        self.all_models = list()

    def load_all_models(self, models):
        for name, (_, model, _) in models.items():
            checkpoint_path = os.path.join(self.run_paths["path_ckpts_train"], name)
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint_manager = tf.train.CheckpointManager(
                checkpoint, directory=checkpoint_path, max_to_keep=5)
            # restore checkpoints
            checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
            if checkpoint_manager.latest_checkpoint:
                logging.info("Restored {} from {}".format(name, checkpoint_manager.latest_checkpoint))
                self.all_models.append(model)

            else:
                logging.info("{} model not loaded".format(name))
                return False
        return True

    def ensemble_predictions(self, test_images):
        # make predictions for all models
        predictions = [model.predict(test_images) for model in self.all_models]
        predictions = np.array(predictions)
        # sum across ensemble members
        summed = np.sum(predictions, axis=0)
        return summed

