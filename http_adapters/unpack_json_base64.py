import base64
from io import BytesIO

import numpy as np
from fastapi import FastAPI, HTTPException
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
    name="UnpackJSON",
    ray_actor_options={
        "num_gpus": 0.0,
        "num_cpus": 0.05,
        "memory": 100,
        "runtime_env": {
            "image": "rayproject/ray:2.7.0-py310",
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
class UnpackJSONbase64:
    @app.post("/")
    async def base64_to_ndarray(self, json_img_base64):
        # convert to json
        image_base64 = await json_img_base64.json()

        try:
            # convert from base64 to pillow object
            image_bytes = base64.b64decode(image_base64)
            image_pillow = Image.open(BytesIO(image_bytes))

            # convert to np array
            image_np = np.array(image_pillow)

            return image_np

        except:
            raise HTTPException(status_code=400, detail="Invalid image format")
