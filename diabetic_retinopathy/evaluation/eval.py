import tensorflow as tf
from evaluation.metrics import ConfusionMatrix

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
    def __init__(self, model, ds_test, ds_info, run_paths):
        self.model = model.load_weights(filepath=run_paths["path_ckpts_train"])
        self.ds_test = ds_test
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.epoch = 0
        self.confusion_matrix = None
        self.confusion_metrics = ConfusionMatrix()
        # self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        # self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def test_step(self, images, labels, num_classes):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        self.confusion_matrix = tf.math.confusion_matrix(labels, predictions, num_classes)
        self.confusion_metrics.update_state(predictions, labels, self.epoch)

    def evaluate(self):
        for test_images, test_labels in self.ds_test:
            self.epoch += 1
            self.test_step(test_images, test_labels, self.ds_info.features["label"].num_classes)
        return self.confusion_matrix

