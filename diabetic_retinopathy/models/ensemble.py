import tensorflow as tf
import os


class Ensemble(object):
    def __init__(self, models, run_paths, learning_rate):
        self.models = models
        self.run_paths = run_paths
        self.all_models = list()
        self.ensemble_model = None
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # load models from file
    def load_all_models(self):
        for name, model in self.models.items():
            checkpoint_path = os.path.join(self.run_paths["path_ckpts_train"], name)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
                                             optimizer=optimizer, model=model)
            checkpoint_manager = tf.train.CheckpointManager(
                checkpoint, directory=checkpoint_path, max_to_keep=5)
            if checkpoint_manager.latest_checkpoint:
                print("Restored {} from {}".format(name, checkpoint_manager.latest_checkpoint))
                self.all_models.append(model)
            else:
                print("No model loaded")

    # define stacked model from multiple member input models
    def define_stacked_model(self, n_classes=2, dense_units=4):
        self.load_all_models()
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
