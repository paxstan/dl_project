import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Trainer(object):
    def __init__(self, model, x, y, epoch):
        self.model = model
        self.x = x
        self.y = y
        self.epoch = epoch

    def train(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = self.model.fit(np.array(self.x), np.array(self.y), epochs=self.epoch)
        self.model.save_weights('model_lstm.h5')
        metrics = history.history
        plt.plot(history.epoch, metrics['loss'], metrics['accuracy'])
        plt.legend(['loss', 'accuracy'])
        plt.savefig("learning-lstm.png")
        plt.show()
        plt.close()
