import grpc
from keras_image_helper import create_preprocessor
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from io import BytesIO
# from flask import Flask, request, jsonify
from proto import np_to_protobuf
from PIL import Image
import os
import json

from fastapi import FastAPI, File, UploadFile

ML_SERVER_HOST = os.getenv('TF_SERVING_HOST','localhost:8500')
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
preprocessor = create_preprocessor('xception', target_size=TARGET_SIZE)

def preprocess_data(preprocessor, img):
    X= preprocessor.convert_to_tensor(img)
    return X

def prepare_request(X):

    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = 'mango-model'
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs['input_2'].CopyFrom(np_to_protobuf(X))
    return pb_request


def prepare_response(pb_response, classes):
    return dict(zip(classes, pb_response.outputs['dense_1'].float_val))


def predict(preprocessor, img, classes):

    X = preprocess_data(preprocessor, img)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response, classes)
    return response

# app = Flask('gateway')
app = FastAPI()

# @app.route('/predict', methods=['POST'])
@app.post('/predict')
async def predict_endpoint(file: UploadFile = File(...)):
    print(file, flush=True)
    image_bytes = await file.read()
    print(image_bytes, flush=True)
    img = Image.open(BytesIO(image_bytes))
    result = predict(preprocessor, img, CLASSES)
    print(result, flush=True)
    return json.dumps(result)
    """
        def predict_endpoint():
        image_stream = request.files.get('mango', '')
        print(image_stream, flush=True)
        img = Image.open(image_stream)
        result = predict(preprocessor, img, CLASSES)
        print(result, flush=True)
        return jsonify(result)
    """

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9000)
