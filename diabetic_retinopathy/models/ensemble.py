import gin
import tensorflow as tf
import os
import numpy as np

@gin.configurable
class Ensemble(object):
    def __init__(self, models, run_paths, learning_rate):
        # self.models = models
        self.run_paths = run_paths
        self.all_models = list()
        self.ensemble_model = None
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # load models from file
    def load_all_models(self, models):
        for name, (_, model) in models.items():
            checkpoint_path = os.path.join(self.run_paths["path_ckpts_train"], name)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                             optimizer=optimizer, model=model)
            checkpoint_manager = tf.train.CheckpointManager(
                checkpoint, directory=checkpoint_path, max_to_keep=5)
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            if checkpoint_manager.latest_checkpoint:
                print("Restored {} from {}".format(name, checkpoint_manager.latest_checkpoint))
                self.all_models.append(model)
            else:
                print("{} model not loaded".format(name))
                return False
        return True

    # define stacked model from multiple member input models
    def define_stacked_model(self, n_classes, dense_units):
        # update all layers in all models to not be trainable
        for i in range(len(self.all_models)):
            model = self.all_models[i]
            for layer in model.layers:
                # make not trainable
                layer.trainable = False
                # rename to avoid 'unique layer name' issue
                layer._name = 'ensemble_' + str(i + 1) + '_' + layer.name
        # define multi-headed input
        ensemble_visible = [model.input for model in self.all_models]
        # concatenate merge output from each model
        ensemble_outputs = [model.output for model in self.all_models]
        merge = tf.keras.layers.concatenate(ensemble_outputs)
        hidden = tf.keras.layers.Dense(dense_units, activation='relu')(merge)
        output = tf.keras.layers.Dense(n_classes)(hidden)
        self.ensemble_model = tf.keras.Model(inputs=ensemble_visible, outputs=output)
        # plot graph of ensemble
        tf.keras.utils.plot_model(self.ensemble_model, show_shapes=True,
                                  to_file='/home/RUS_CIP/st180304/st180304/model_graph.png')
        # compile
        self.ensemble_model.compile(loss=self.loss_object, optimizer=self.optimizer)

    def ensemble_prediction(self, test_images, test_labels):
        stack_x = None
        for name, (_, model) in self.models.items():
            # make prediction
            predictions = model(test_images, training=False)
            y_pred = tf.reshape(tf.argmax(predictions, axis=1), shape=(-1, 1))
            # stack predictions into [rows, members, probabilities]
            if stack_x is None:
                stack_x = y_pred
            else:
                stack_x = np.dstack((stack_x, y_pred))
        return stack_x

    # create stacked model input dataset as outputs from the ensemble
    def stacked_dataset(self, test_image):
        stack_x = None
        for name, (_, model) in self.models.items():
            # make prediction
            predictions = model(test_image, training=False)
            # y_pred = tf.reshape(tf.argmax(predictions, axis=1), shape=(-1, 1))
            # stack predictions into [rows, members, probabilities]
            if stack_x is None:
                stack_x = predictions
            else:
                stack_x = np.dstack((stack_x, predictions))
        # flatten predictions to [rows, members x probabilities]
        stack_x = stack_x.reshape((stack_x.shape[0], stack_x.shape[1] * stack_x.shape[2]))
        return stack_x

    # fit a model based on the outputs from the ensemble members
    def fit_stacked_model(self, inputX, inputy):
        # create dataset using ensemble
        stackedX = self.stacked_dataset(members, inputX)
        # fit standalone model
        model = LogisticRegression()
        model.fit(stackedX, inputy)
        return model

    #
    # # make a prediction with the stacked model
    def stacked_prediction(members, model, inputX):
        # create dataset using ensemble
        stackedX = stacked_dataset(members, inputX)
        # make a prediction
        yhat = model.predict(stackedX)
        return yhat
