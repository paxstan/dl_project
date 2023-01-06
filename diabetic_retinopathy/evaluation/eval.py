import tensorflow as tf
import logging
from evaluation.metrics import ConfusionMatrix, BinaryTruePositives


class Evaluation(object):
    """class for evaluation"""
    def __init__(self, model, ds_test, ds_info):
        self.model = model
        self.ds_test = ds_test
        self.ds_info = ds_info
        self.epoch = 0
        self.confusion_metrics = ConfusionMatrix()
        self.true_positive = BinaryTruePositives()
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # @tf.function
    def metric_calculation(self, predictions, labels, num_classes):
        """function to calculate the metrics"""
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        y_pred = tf.reshape(tf.argmax(predictions, axis=1), shape=(-1, 1))
        self.confusion_metrics.update_state(y_pred, labels, num_classes)
        self.true_positive.update_state(y_pred, labels)

    def evaluate(self, ensemble=False):
        """function to evaluate the model against test dataset"""
        for test_images, test_labels in self.ds_test:
            self.epoch += 1
            self.confusion_metrics.reset_state()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.true_positive.reset_state()
            if ensemble:
                predictions = self.model.ensemble_predictions(test_images)
            else:
                predictions = self.model.predict(test_images)

            self.metric_calculation(predictions, test_labels, self.ds_info.features["label"].num_classes)
            template = 'Step {}, Test Loss: {}, Test Accuracy: {}, True Positives: {},\n Confusion matrix: \n{}'
            logging.info(template.format(
                self.epoch,
                self.test_loss.result().numpy(),
                self.test_accuracy.result().numpy() * 100,
                self.true_positive.result().numpy(),
                self.confusion_metrics.result()
            ))


