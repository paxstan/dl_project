import tensorflow as tf
from evaluation.metrics import ConfusionMatrix, BinaryTruePositives


class Evaluation(object):
    def __init__(self, ensemble, ds_test, ds_info):
        self.model = ensemble
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
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        y_pred = tf.reshape(tf.argmax(predictions, axis=1), shape=(-1, 1))
        # y_pred = tf.argmax(predictions, axis=-1)
        self.confusion_metrics.update_state(y_pred, labels, num_classes)
        self.true_positive.update_state(y_pred, labels)

    def evaluate(self, ensemble):
        for test_images, test_labels in self.ds_test:
            self.epoch += 1
            self.confusion_metrics.reset_state()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.true_positive.reset_state()
            if ensemble:
                test_images = [test_images for _ in range(len(self.model.input))]
                predictions = self.model(test_images, training=False)
            else:
                predictions = self.model.ensemble_predictions(test_images)

            self.metric_calculation(predictions, test_labels, self.ds_info.features["label"].num_classes)
            template = 'Step {}, Test Loss: {}, Test Accuracy: {}, True Positives: {},\n Confusion matrix: \n{}'
            print(template.format(
                self.epoch,
                self.test_loss.result().numpy(),
                self.test_accuracy.result().numpy() * 100,
                self.true_positive.result().numpy(),
                self.confusion_metrics.result()
            ))

