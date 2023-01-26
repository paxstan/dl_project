import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback


class Trainer(object):
    def __init__(self, model, optimizer, loss_fn, epoch, train_ds, val_ds, display_cb, wandb_key):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epoch = epoch
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.display_cb = display_cb
        wandb.login(key=wandb_key)
        wandb.init(project='', entity="dl-team-07")

    def train(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)

        history = self.model.fit(
            self.train_ds, validation_data=self.val_ds, callbacks=[self.display_cb, WandbCallback()], epochs=self.epoch)

        # self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # history = self.model.fit(np.array(self.x), np.array(self.y), epochs=self.epoch)
        self.model.save_weights('model_transformer.h5')
        metrics = history.history
        plt.plot(history.epoch, metrics['loss'], metrics['accuracy'])
        plt.legend(['loss', 'accuracy'])
        plt.savefig("learning-transformer.png")
        plt.show()
        plt.close()
