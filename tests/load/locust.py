"""
Need:
    pip install locust

Run:    
    DATASET_PATH="/.../images" locust -f locustfile.py \ 
    --headless -u NUM_CLIENTES -r RATE_CLIENTS_PER_SECOUND 
    --run-time EXECUTION_TIME -H URL_API --csv CSV_NAME    
"""
from locust import HttpUser, task, between
import os
import random

class UserBehavior(HttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def upload_image(self):

        folder_path = os.getenv("DATASET_PATH")

        supported_formats = (".jpg", ".jpeg", ".png")

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_formats)]        

        selected_file = random.choice(files)

        file_path = os.path.join(folder_path, selected_file)

        with open(file_path, "rb") as f:
            self.client.post("/uploadfile", files={"file": f})
