import tensorflow as tf
from tensorflow import keras
import sys

MODEL_PATH = "./models/best-model-v4.37_0.981.h5"
OUTPUT_FOLDER = "../production/tf-serving/"
MODEL_NAME = "mango-model"
OUTPUT_MODEL_PATH = OUTPUT_FOLDER + MODEL_NAME

def convert_model(model_path, output_model_path):
    model = keras.models.load_model(model_path)
    tf.saved_model.save(model, output_model_path)

if __name__ == '__main__':

    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    output_model_path = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_MODEL_PATH
    convert_model(model_path, output_model_path)
  