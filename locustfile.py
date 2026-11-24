"""Load test /predict using a real sample image (not /health)."""
from pathlib import Path
from locust import HttpUser, task, between

IMAGE = (Path(__file__).parent / 'data/test/digit_3.png').read_bytes()

class DigitUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict_digit(self):
        self.client.post('/predict', files={'file': ('digit_3.png', IMAGE, 'image/png')},
                         name='/predict')
