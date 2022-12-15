from locust import task
from locust import between
from locust import HttpUser

IMAGE_NAME = "IMG_20210630_102920.jpg"
IMAGE_PATH = "./test_data/" + IMAGE_NAME

class PerformanceUser(HttpUser):
    """
    Usage:
        Start locust load testing client with:

            locust -H http://localhost:8080

        Open browser at http://localhost:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    @task
    def predict(self):
        with open(IMAGE_PATH, 'rb') as image_file:
            files = {'file': image_file}
            self.client.post("/predict", files=files)

    wait_time = between(0.100, 0.250)



