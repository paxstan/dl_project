# Speech Recognition on Android with Wav2Vec2

## Introduction

Facebook AI's [wav2vec 2.0](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) is one of the leading models in speech recognition. It is also available in the [Huggingface Transformers](https://github.com/huggingface/transformers) library.

In this demo app, we'll show how to quantize, trace, and optimize the wav2vec2 model, powered by the newly released torchaudio 0.9.0, and how to use the converted model on an Android demo app to perform speech recognition.

## Prerequisites

* PyTorch 1.10.0 and torchaudio 0.10.0 (Optional)
* Python 3.8 (Optional)
* Android Pytorch library org.pytorch:pytorch_android_lite:1.10.0
* Android Studio 4.0.1 or later


