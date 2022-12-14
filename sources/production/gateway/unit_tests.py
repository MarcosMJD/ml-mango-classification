from starlette.testclient import TestClient
from gateway import app

TEST_IMAGE_PATH = './TEST_DATA/IMG_20210630_102920.jpg'

def test_root(client):

    response = client.get('/')
    assert response.text == '<a href="/docs">Click here to access API</a>'

def test_predict_endpoint_ok(client, image_path):

    files = {'file': open(image_path, 'rb')}
    response = client.post('/predict', files=files)
    assert response.status_code == 200
    assert response.json() == {
        'Anwar Ratool': 1.0,
        'Chaunsa (Black)': 0.0,
        'Chaunsa (Summer Bahisht)': 0.0,
        'Chaunsa (White)': 5.0123284672731474e-37,
        'Dosehri': 0.0,
        'Fajri': 2.4020681443702053e-25,
        'Langra': 8.949817257817495e-34,
        'Sindhri': 5.200761237684644e-35
    }
    return 'ok'

def test_predict_endpoint_bad_file(client):

    files = {'file': open(__file__, 'rb')}
    response = client.post('/predict', files=files)
    assert response.status_code == 422


if __name__ == "__main__":

    client = TestClient(app)
    test_predict_endpoint_ok(client,TEST_IMAGE_PATH)
    test_predict_endpoint_bad_file(client)
    test_root(client)