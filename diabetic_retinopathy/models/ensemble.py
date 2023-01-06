import gin
import tensorflow as tf
import os
import numpy as np


@gin.configurable
class Ensemble(object):
    def __init__(self, run_paths):
        # self.models = models
        self.run_paths = run_paths
        self.all_models = list()
        self.ensemble_model = None
        # self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # load models from file
    def load_all_models(self, models, add_model=True):
        for name, (_, model, _) in models.items():
            checkpoint_path = os.path.join(self.run_paths["path_ckpts_train"], name)
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint_manager = tf.train.CheckpointManager(
                checkpoint, directory=checkpoint_path, max_to_keep=5)
            checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
            if checkpoint_manager.latest_checkpoint:
                print("Restored {} from {}".format(name, checkpoint_manager.latest_checkpoint))
                if add_model:
                    self.all_models.append(model)
                else:
                    self.ensemble_model = model
            else:
                print("{} model not loaded".format(name))
                return False
        return True

    def ensemble_predictions(self, test_images):
        # make predictions
        predictions = [model.predict(test_images) for model in self.all_models]
        predictions = np.array(predictions)
        # sum across ensemble members
        summed = np.sum(predictions, axis=0)
        return summed

