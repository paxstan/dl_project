import gin
import torch
from torch import Tensor
from torch.utils.mobile_optimizer import optimize_for_mobile
import torchaudio
from torchaudio.models.wav2vec2.utils.import_huggingface import import_huggingface_model
from transformers import Wav2Vec2ForCTC


# Wav2vec2 model emits sequences of probability (logits) distributions over the characters
# The following class adds steps to decode the transcript (best path)
class SpeechRecognizer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.labels = [
            "<s>", "<pad>", "</s>", "<unk>", "|", "E", "T", "A", "O", "N", "I", "H", "S",
            "R", "D", "L", "U", "M", "W", "C", "F", "G", "Y", "P", "B", "V", "K", "'", "X",
            "J", "Q", "Z"]

    def forward(self, waveforms: Tensor) -> str:
        """Given a single channel speech data, return transcription.

        Args:
            waveforms (Tensor): Speech tensor. Shape `[1, num_frames]`.

        Returns:
            str: The resulting transcript
        """
        logits, _ = self.model(waveforms)  # [batch, num_seq, num_label]
        best_path = torch.argmax(logits[0], dim=-1)  # [num_seq,]
        prev = ''
        hypothesis = ''
        for i in best_path:
            char = self.labels[i]
            if char == prev:
                continue
            if char == '<s>':
                prev = ''
                continue
            hypothesis += char
            prev = char
        return hypothesis.replace('|', ' ')


@gin.configurable
class ModelToMobileConvertor(object):
    def __init__(self, run_paths, audio_path):
        # Load Wav2Vec2 pretrained model from Hugging Face Hub
        self.model = Wav2Vec2ForCTC.from_pretrained(run_paths['path_saved_model_train'])
        self.mobile_model_path = "{}/wav2vec2_mobile.ptl".format(run_paths['path_results'])
        self.audio_path = audio_path

    def convert_model(self):
        """
        Converts the model to mobile compatible and saves it

        """
        # Convert the model to torchaudio format, which supports TorchScript.
        model = import_huggingface_model(self.model)

        # Remove weight normalization which is not supported by quantization.
        model.encoder.transformer.pos_conv_embed.__prepare_scriptable__()

        model = model.eval()

        # Attach decoder
        model = SpeechRecognizer(model)

        # Apply quantization / script / optimize for motbile
        quantized_model = torch.quantization.quantize_dynamic(
            model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        scripted_model = torch.jit.script(quantized_model)
        optimized_model = optimize_for_mobile(scripted_model)
        self.test_model(optimized_model)
        optimized_model._save_for_lite_interpreter(self.mobile_model_path)
        print(f"Mobile model saved at {self.mobile_model_path}")

    def test_model(self, optimized_model):
        """
        Test the mobile model with a local audio file
        Args:
            optimized_model: mobile converted model
            audio_path: path to the local file

        Returns:
            prints the result
        """
        # Sanity check
        waveform, _ = torchaudio.load(self.audio_path)
        print('Result:', optimized_model(waveform))
