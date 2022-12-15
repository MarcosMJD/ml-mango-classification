import requests

GATEWAY = 'http://localhost:8080/predict'

IMAGE_NAME = "IMG_20210630_102920.jpg"
IMAGE_PATH = "./test_data/" + IMAGE_NAME

if __name__ == '__main__':
    
    with open(IMAGE_PATH, 'rb') as image_file:

        files = {'file': image_file}
        result = requests.post(GATEWAY, files=files)
        result = result.json()

        print(result)