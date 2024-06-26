import gin
import tensorflow as tf
import logging
import wandb
from wandb.keras import WandbCallback
import numpy as np
import os


@gin.configurable
class Trainer(object):
    """Class for training models"""
    def __init__(self, model, model_name, ds_train, ds_val, ds_info, run_paths
                 , total_steps, log_interval, ckpt_interval, learning_rate, batch_size, wandb_key):
        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.learning_rate = learning_rate
        self.wandb_key = wandb_key
        self.batch_size = batch_size

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

        # Checkpoint Manager
        self.checkpoint_path = os.path.join(run_paths["path_ckpts_train"], model_name)
        if not os.path.isdir(self.checkpoint_path):
            os.makedirs(self.checkpoint_path, exist_ok=True)
        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                              optimizer=self.optimizer, model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.checkpoint_path, max_to_keep=3)

        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        config = {
            "learning_rate": learning_rate,
            "epochs": total_steps,
            "batch_size": batch_size,
            "log_step": log_interval,
            "val_log_step": log_interval,
            "architecture": "CNN",
            "dataset": "IDRID"
        }
        wandb.login(anonymous="allow", key=self.wandb_key)
        self.run = wandb.init(project='idrid-test', config=config)

    @tf.function
    def train_step(self, images, labels):
        """function for training the model"""
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        """function for validating the trained model"""
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.val_loss(t_loss)
        self.val_accuracy(labels, predictions)

    def train(self):
        for idx, (images, labels) in enumerate(self.ds_train):
            self.checkpoint.step.assign_add(1)
            self.train_step(images, labels)

            if int(self.checkpoint.step) % self.log_interval == 0:

                # Reset test metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(int(self.checkpoint.step),
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))

                # ⭐: log metrics using wandb.log
                log = {'loss': np.mean(self.train_loss.result()),
                       'acc': float(self.train_accuracy.result()),
                       'val_loss': np.mean(self.val_loss.result()),
                       'val_acc': float(self.val_accuracy.result()),
                       'step': int(self.checkpoint.step)}

                wandb.log(log)

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            if int(self.checkpoint.step) % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.checkpoint_path}.')
                self.checkpoint_manager.save()

            if int(self.checkpoint.step) % self.total_steps == 0:
                logging.info(f'Finished training after {int(self.checkpoint.step)} steps.')
                # Save final checkpoint
                self.checkpoint_manager.save()
                return self.val_accuracy.result().numpy()
