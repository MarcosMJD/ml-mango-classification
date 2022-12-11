from typing import Union, List
from io import BytesIO
from PIL import Image
import os
import json
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

class Item_post(BaseModel):
    param_1: str
    param_2: int=0
    param_3: Union[float,None] = None
    param_3: Union[List[str],None] = None

app = FastAPI()

@app.get('/endpoint_get/{path}')
async def endpoint_get(path, param_1: str, param_2:int = 0, param_3: Union[float,None] = None):
    answer = {
        "path": path,
        "param_1": param_1,
        "param_2": param_2,
        "param_3": param_3
    }
    return answer

@app.post('/endpoint_post/{path}')
async def endpoint_post(path, item: Item_post, query_1:str="", query_2:int=0):
    answer = {
        'path': path,
        'query_1': query_1,
        'query_2': query_2,
        **item.dict()
    }
    return answer


@app.post('/image')
async def predict_endpoint(file: UploadFile = File(...)):
    print(file, flush=True)
    image_bytes = await file.read()
    print(image_bytes, flush=True)
    img = Image.open(BytesIO(image_bytes))
    
    print(img, flush=True)
    return "ok"
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
