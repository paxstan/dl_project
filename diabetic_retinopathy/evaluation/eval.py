import tensorflow as tf
from evaluation.metrics import ConfusionMatrix, BinaryTruePositives

#
# def evaluate(model, ds_test, ds_info, run_paths):
#     model.load_weights(filepath=run_paths["path_ckpts_train"])
#     # test_images, test_labels = ds_test.map(lambda x, y: (x, y))
#     # test_pred = model.predict(test_images)
#     performance_cbk = PerformanceVisualizationCallback(
#         model=model,
#         ds_test=ds_test,
#         image_dir='performance_vizualizations')
#     performance_cbk.on_epoch_end()
#     return


class Evaluation(object):
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
    def test_step(self, images, labels, num_classes):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        y_pred = tf.reshape(tf.argmax(predictions, axis=1), shape=(-1, 1))
        # y_pred = tf.argmax(predictions, axis=-1)
        self.confusion_metrics.update_state(y_pred, labels, num_classes)
        self.true_positive.update_state(y_pred, labels)

    def evaluate(self):
        for test_images, test_labels in self.ds_test:
            self.epoch += 1
            self.confusion_metrics.reset_state()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            self.true_positive.reset_state()
            X = [test_images for _ in range(len(self.model.input))]
            self.test_step(X, test_labels, self.ds_info.features["label"].num_classes)
            template = 'Step {}, Test Loss: {}, Test Accuracy: {}, True Positives: {},\n Confusion matrix: \n{}'
            print(template.format(
                self.epoch,
                self.test_loss.result().numpy(),
                self.test_accuracy.result().numpy() * 100,
                self.true_positive.result().numpy(),
                self.confusion_metrics.result()
            ))
        # return self.confusion_metrics.result()

