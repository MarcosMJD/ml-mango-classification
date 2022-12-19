import tensorflow as tf
from tensorflow import keras
import sys
from argparse import ArgumentParser

MODEL_PATH = "./models/best-model-v4.37_0.981.h5"
OUTPUT_FOLDER = "../production/tf-serving/"
MODEL_NAME = "mango-model"

def convert_model(model_path, output_model_path):
    model = keras.models.load_model(model_path)
    tf.saved_model.save(model, output_model_path)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument(
        '--modelpath',
        default=MODEL_PATH,
        help='Path of the model to convert, including its name. e.j. "./models/model.tf5"')
    
    parser.add_argument(
        '--outputdir',
        default=OUTPUT_FOLDER,
        help='Directory where to save the model. E.g. "../my-model/')

    args = parser.parse_args()

    model_path = args.modelpath
    output_model_path = args.outputdir + MODEL_NAME
    convert_model(model_path, output_model_path)
  