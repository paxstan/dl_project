import torch
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML


class Evaluation(object):
    def __init__(self, model, processor, dataset):
        self.model = model
        self.processor = processor
        self.dataset = dataset

    def evaluate(self):
        results = self.dataset.map(self.map_to_result, remove_columns=self.dataset.column_names)
        show_random_elements(results)

    def map_to_result(self, batch):
        with torch.no_grad():
            input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
            logits = self.model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_str"] = self.processor.batch_decode(pred_ids)[0]
        batch["text"] = self.processor.decode(batch["labels"], group_tokens=False)

        return batch


def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    display(HTML(df.to_html()))