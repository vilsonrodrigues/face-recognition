from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ray import serve

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@serve.deployment(
    name="UploadFile",
    ray_actor_options={
        "num_gpus": 0.0,
        "num_cpus": 0.05,
        "memory": 100,
        "runtime_env": {
            "image": "rayproject/ray:2.7.1-py310",
            # "run_options": ["--cap-drop SYS_ADMIN","--log-level=debug"]
        },
    },
    health_check_period_s=10,
    health_check_timeout_s=30,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 16,
        "initial_replicas": 4,
    },
)
@serve.ingress(app)
class UploadFileToNdarray:
    @app.post("/")
    async def upload_image(
        file: UploadFile = File(description="Image with faces", media_type="image/*")
    ):
        image_bytes = await file.read()

        try:
            image_pillow = Image.open(BytesIO(image_bytes))

            image_np = np.array(image_pillow)

            return image_np

        except:
            raise HTTPException(status_code=400, detail="Invalid image format")
