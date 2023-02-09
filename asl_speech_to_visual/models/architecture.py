from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2100h(object):
    """
    Class to load Wav2Vec2 model
    """
    def __init__(self):
        # get processor from pretrained model
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")
        # get pretrained model from hugging face hub
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-100h",
                                                    ctc_loss_reduction="mean",
                                                    pad_token_id=self.processor.tokenizer.pad_token_id)

        # freeze the stack of CNN layers which is used for feature extraction
        self.model.freeze_feature_encoder()

    def load_from_checkpoint(self, run_paths):
        """
        Load from checkpoint to continue training
        Args:
            run_paths: dictionary of run path
        """
        self.model = Wav2Vec2ForCTC.from_pretrained(run_paths['path_ckpts_train'])

    def load_from_saved_model(self, run_paths):
        """
        Loads saved model for evaluation
        Args:
            run_paths:

        """
        self.model = Wav2Vec2ForCTC.from_pretrained(run_paths['path_saved_model_train']).cuda()
