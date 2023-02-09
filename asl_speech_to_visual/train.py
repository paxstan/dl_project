import gin
from transformers import TrainingArguments, Trainer
from evaluation.metrics import WerMetricClass
from input_pipeline.preprocessing import DataCollatorCTCWithPadding
import wandb


@gin.configurable
class Train(object):
    def __init__(self, model, processor, ds_train, ds_val, run_paths,
                 epochs, learning_rate, weight_decay, wandb_key, wandb_project):
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        wer_metric = WerMetricClass(processor)
        self.run_paths = run_paths
        wandb.login(anonymous="allow", key=wandb_key)
        wandb.init(project=wandb_project, entity="dl-team-07")
        training_args = TrainingArguments(
            output_dir=run_paths['path_ckpts_train'],
            group_by_length=True,
            per_device_train_batch_size=8,
            evaluation_strategy="steps",
            num_train_epochs=epochs,
            fp16=True,
            gradient_checkpointing=True,
            save_steps=500,
            eval_steps=500,
            logging_steps=500,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to=["wandb"]
        )
        self.trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=wer_metric.compute_metrics,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            tokenizer=processor.feature_extractor,
        )

    def train(self):
        """
        To train the model
        """
        self.trainer.train()
        self.trainer.save_model(output_dir=self.run_paths['path_saved_model_train'])
