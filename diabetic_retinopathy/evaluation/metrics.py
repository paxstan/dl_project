import os
import tensorflow as tf


# from keras.callbacks import Callback
# import matplotlib.pyplot as plt
# import numpy as np
# from scikitplot.metrics import plot_confusion_matrix, plot_roc


class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.confusion_matrix = None
        # self.image_dir = "/home/paxstan/Documents/Uni/DL_Lab"

    def update_state(self, y_true, y_pred, num_classes):
        self.confusion_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes)

        # plot and save confusion matrix
        # fig, ax = plt.subplots(figsize=(16, 12))
        # plot_confusion_matrix(y_true, y_pred, ax=ax)
        # fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))

    def result(self):
        return self.confusion_matrix

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.confusion_matrix = None


class BinaryTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)
