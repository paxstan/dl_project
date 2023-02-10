import gin
import logging
from absl import app, flags
from utils import utils_params, utils_misc
from input_pipeline.asl_dataset import LoadDataset
from evaluation.eval import Evaluation
from train import Train
from lite_model_convertor import ModelToMobileConvertor
from models.architecture import Wav2Vec2100h

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_boolean('saved_model', False, 'Specify whether to train or evaluate a model.')
flags.DEFINE_string('runId', "", 'Specify path to the run directory.')
flags.DEFINE_string('convert_model', "", 'Specify whether to convert the model to mobile compatible.')


def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder(FLAGS.runId)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # pretrained model object
    wav2vec2 = Wav2Vec2100h()

    # load dataset from parquet files
    load_dataset = LoadDataset(processor=wav2vec2.processor)
    ds_train, ds_val, ds_test = load_dataset.load()

    if FLAGS.train:
        # load from checkpoint
        if FLAGS.saved_model and FLAGS.runId:
            wav2vec2.load_from_checkpoint(run_paths)

        # finetune the model
        train = Train(wav2vec2.model, wav2vec2.processor, ds_train, ds_val, run_paths)
        train.train()
    else:
        # load saved model for evaluation
        wav2vec2.load_from_saved_model(run_paths)

        # evaluate the model
        evaluation = Evaluation(wav2vec2.model, wav2vec2.processor, ds_test, run_paths)
        evaluation.evaluate()

        if FLAGS.convert_model:
            # convert the model to mobile compatible model
            mobile_model = ModelToMobileConvertor(run_paths)
            mobile_model.convert_model()


if __name__ == "__main__":
    app.run(main)
