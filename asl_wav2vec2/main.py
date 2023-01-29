import gin
import logging
from absl import app, flags
from utils import utils_params, utils_misc
from input_pipeline.asl_dataset import LoadDataset
from input_pipeline.preprocessing import DataCollatorCTCWithPadding
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from evaluation.metrics import WerMetricClass
from evaluation.eval import Evaluation
import wandb

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('runId', "", 'Specify path to the run directory.')


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder(FLAGS.runId)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")

    load_dataset = LoadDataset(processor=processor)
    ds_train, ds_val, ds_test = load_dataset.load()

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    repo_name = "/home/RUS_CIP/st180304/st180304/libri_checkpoint-a"
    wer_metric = WerMetricClass(processor)

    if FLAGS.train:

        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-100h",
                                               ctc_loss_reduction="mean",
                                               pad_token_id=processor.tokenizer.pad_token_id, )
        model.freeze_feature_encoder()

        wandb.login(anonymous="allow", key="8b5621f60202d49f7fa98ffafcb02ebbe4a3a314")

        wandb.init(project="asl_wav2vec2", entity="dl-team-07")

        training_args = TrainingArguments(
            output_dir=repo_name,
            group_by_length=True,
            per_device_train_batch_size=8,
            evaluation_strategy="steps",
            num_train_epochs=30,
            fp16=True,
            gradient_checkpointing=True,
            save_steps=500,
            eval_steps=500,
            logging_steps=500,
            learning_rate=1e-4,
            weight_decay=0.005,
            warmup_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to=["wandb"]
        )
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=wer_metric.compute_metrics,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            tokenizer=processor.feature_extractor,
        )

        trainer.train()

        trainer.save_model(output_dir="/home/RUS_CIP/st180304/st180304/librispeech_save_model-a")

    else:
        checkpoint_dir = "/home/paxstan/Desktop/libri_checkpoint-a/checkpoint-500"
        model = Wav2Vec2ForCTC.from_pretrained(checkpoint_dir).cuda()
        evaluation = Evaluation(model, processor, ds_test)
        evaluation.evaluate()


if __name__ == "__main__":
    app.run(main)
