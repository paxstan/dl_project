import torch
import random
import pandas as pd


class Evaluation(object):
    """
    Class for Evaluation
    """
    def __init__(self, model, processor, dataset, run_paths):
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.result_path = "{}/model_result.csv".format(run_paths['path_results'])

    def evaluate(self):
        """
        To evaluate the saved model
        """
        results = self.dataset.map(self.map_to_result, remove_columns=self.dataset.column_names)
        show_random_elements(results, self.result_path)

    def map_to_result(self, batch):
        """
        To predict the batch of data using saved model
        """
        with torch.no_grad():
            input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
            logits = self.model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_str"] = self.processor.batch_decode(pred_ids)[0]
        batch["text"] = self.processor.decode(batch["labels"], group_tokens=False)

        return batch


def show_random_elements(dataset, result_path, num_examples=10):
    """
    Function to pick random transcripts from the dataset
    Args:
        dataset: test dataset
        result_path: path to save csv result
        num_examples: number of transcript to consider

    Returns:
        saves the result dataframe in a local csv file
    """
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset) - 1)
        while pick in picks:
            pick = random.randint(0, len(dataset) - 1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    df.to_csv(result_path)
    print(f"Model result saved at {result_path}")
