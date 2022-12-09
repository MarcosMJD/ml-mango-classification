import grpc
from keras_image_helper import create_preprocessor
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from proto import np_to_protobuf
import numpy as np
import os

ML_SERVER_HOST = os.getenv('TF_SERVING_HOST','localhost:8500')
IMAGE_NAME = "IMG_20210630_102920.jpg"
IMAGE_PATH = "./" + IMAGE_NAME
TARGET_SIZE = (299,299)

CLASSES =  [
    'Anwar Ratool',
    'Chaunsa (Black)',
    'Chaunsa (Summer Bahisht)',
    'Chaunsa (White)',
    'Dosehri',
    'Fajri',
    'Langra',
    'Sindhri'
]

channel = grpc.insecure_channel(ML_SERVER_HOST)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

def preprocess_data(file_path, target_size):

    # Note: from_path expands one dimension on axis 0 the resulting np array with one image
    preprocessor = create_preprocessor('xception', target_size=target_size)
    X = preprocessor.from_path(file_path)
    return X

def prepare_request(X):

    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = 'mango-model'
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs['input_2'].CopyFrom(np_to_protobuf(X))
    return pb_request


def prepare_response(classes, pb_response):
    return dict(zip(classes, pb_response.outputs['dense_1'].float_val))


def predict(X:np.array, classes)->np.array:

    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(classes, pb_response)
    return response

if __name__ == '__main__':

    X = preprocess_data(IMAGE_PATH, TARGET_SIZE)
    preds = predict(X, CLASSES)
    print(preds)



