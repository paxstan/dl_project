import gin
import logging
import tensorflow as tf
from absl import app, flags
from utils import utils_params, utils_misc
from wav2vec2 import CTCLoss
from models.architecture import wav2vec2_tf
from input_pipeline.asl_dataset import load, LoadDataset
from input_pipeline.preprocessing import DataCollatorCTCWithPadding
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from evaluation.metrics import WerMetricClass

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('runId', "", 'Specify path to the run directory.')
LEARNING_RATE = 5e-5
AUDIO_MAXLEN = 246000
LABEL_MAXLEN = 256
BATCH_SIZE = 2


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder(FLAGS.runId)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-100h")
    model.freeze_feature_encoder()

    load_dataset = LoadDataset(processor=processor)
    dataset = load_dataset.load()
    dataset = dataset.map(load_dataset.prepare_dataset, num_proc=4)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    repo_name = "/home/paxstan/Documents/Uni/DL_Lab/dl-lab-22w-team07/asl_wav2vec2/checkpoint"
    wer_metric = WerMetricClass(processor)

    training_args = TrainingArguments(
        output_dir=repo_name,
        group_by_length=True,
        per_device_train_batch_size=8,
        evaluation_strategy="steps",
        num_train_epochs=30,
        # fp16=True,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2
    )
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=wer_metric.compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=processor.feature_extractor,
    )

    trainer.train()


if __name__ == "__main__":
    app.run(main)
