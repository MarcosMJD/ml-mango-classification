import requests

GATEWAY = 'http://localhost:8080/predict'

TEST_IMAGE = 'IMG_20210630_102920.jpg'

if __name__ == '__main__':
    
    files = {'mango': open(TEST_IMAGE, 'rb')}
    result = requests.post(GATEWAY, files=files)
    result = result.json()

    print(result)