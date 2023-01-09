import gin
import logging
import numpy as np
from absl import app, flags
from input_pipeline import datasets
from models import architectures
from train import Trainer
from evaluation import eval


def main(argv):
    # setup pipeline
    tokenizer, string_max_length, vocab_size, x, y = datasets.load()
    model = architectures.lstm_model(string_max_length, vocab_size)
    epoch = 300
    train = Trainer(model, x, y, epoch)
    train.train()
    eval.predict_word(train.model, "", string_max_length, vocab_size)


if __name__ == "__main__":
    app.run(main)
