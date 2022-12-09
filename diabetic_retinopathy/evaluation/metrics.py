import os
import tensorflow as tf

# from keras.callbacks import Callback
# import matplotlib.pyplot as plt
# import numpy as np
# from scikitplot.metrics import plot_confusion_matrix, plot_roc


class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init(self, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, epoch, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        # plot and save confusion matrix
        # fig, ax = plt.subplots(figsize=(16, 12))
        # plot_confusion_matrix(y_true, y_pred, ax=ax)
        # fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)


# class PerformanceVisualizationCallback(tf.keras.callbacks.Callback):
#     def __init__(self, model, ds_test, image_dir):
#         super().__init__()
#         self.model = model
#         self.ds_test = ds_test
#         os.makedirs(image_dir, exist_ok=True)
#         self.image_dir = image_dir
#
#     def on_epoch_end(self, epoch, logs={}):
#         y_pred = np.asarray(self.model.predict(self.ds_test[0]))
#         y_true = self.ds_test[1]
#         y_pred_class = np.argmax(y_pred, axis=1)
#
#         # plot and save confusion matrix
#         fig, ax = plt.subplots(figsize=(16, 12))
#         plot_confusion_matrix(y_true, y_pred_class, ax=ax)
#         fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))
#
#         # plot and save roc curve
#         fig, ax = plt.subplots(figsize=(16, 12))
#         plot_roc(y_true, y_pred, ax=ax)
#         fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))

