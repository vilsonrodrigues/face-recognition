import asyncio
import base64
from io import BytesIO
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ray import serve
from ray.serve.handle import DeploymentHandle

from auth.oauth2 import verify_token
from processing.utils import check_img_channels
from schemas.responses import FaceResponse

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@serve.deployment(
    name="RetrievalHandler",
    ray_actor_options={
        "num_gpus": 0.0,
        "num_cpus": 1.0,
        "memory": 4096,
    },
    health_check_period_s=10,
    health_check_timeout_s=30,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 2,
        "initial_replicas": 1,
    },
)
@serve.ingress(app)
class PipelineRetrieval:
    def __init__(
        self,
        face_det: DeploymentHandle,
        face_emb: DeploymentHandle,
        face_postp: DeploymentHandle,
        neural_search: DeploymentHandle,
    ):
        self._face_det: DeploymentHandle = face_det.options(
            use_new_handle_api=True,
        )
        self._face_emb: DeploymentHandle = face_emb.options(
            use_new_handle_api=True,
        )
        self._face_postp: DeploymentHandle = face_postp.options(
            use_new_handle_api=True,
        )
        self._face_postp_gen: DeploymentHandle = face_postp.options(
            use_new_handle_api=True,
            stream=True,
        )
        self._neural_search: DeploymentHandle = neural_search.options(
            use_new_handle_api=True,
        )

    async def _route(self, image_np: np.ndarray):
        relative_boxes_ref = self._face_det.predict.remote(image_np)

        relative_boxes = await relative_boxes_ref

        # if no face detected, return a empty list
        if relative_boxes.size == 0:
            return {
                "payloads": [],
                "boxes": [],
            }

        else:
            # get rescaled boxes
            boxes_rescaled = self._face_postp.apply_rescale_boxes.remote(
                relative_boxes, image_np
            )

            # get a faces generator
            faces_generator = self._face_postp_gen.apply_crop_faces.remote(
                boxes_rescaled, image_np
            )

            # async call to face emb and neural search deployment
            tasks = [
                self._neural_search.search.remote(self._face_emb.predict.remote(face))
                async for face in faces_generator
            ]

            # gather payloads
            payloads = asyncio.gather(*tasks)

            return {
                "payloads": await payloads,
                "boxes": await boxes_rescaled,
            }

    @app.post(
        "/uploadfile",
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_token)],
    )
    async def uploadfile_router(
        self,
        file: Annotated[UploadFile, File(description="Image", media_type="image/*")],
    ):
        """Upload image. Retrieve payloads"""
        image_bytes = await file.read()
        try:
            image_pillow = Image.open(BytesIO(image_bytes))
            image_np = np.array(image_pillow)
            if check_img_channels(image_np) == False:
                raise ValueError("Image has invalid dimensions")
        except:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid image format"
            )
        try:
            response = await self._route(image_np)
            return response
        except Exception as e:
            raise HTTPException(status.HTTP_424_FAILED_DEPENDENCY, "Retrive failed")

    @app.post(
        "/base64", status_code=status.HTTP_200_OK, dependencies=[Depends(verify_token)]
    )
    async def base64_router(
        self, image_base64: Annotated[str, Form(..., description="Base64 Image")]
    ):
        """Base64 image. Retrieve payloads"""
        try:
            image_bytes = base64.b64decode(image_base64)
            image_pillow = Image.open(BytesIO(image_bytes))
            image_np = np.array(image_pillow)
            if check_img_channels(image_np) == False:
                raise ValueError("Image has invalid dimensions")
        except:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, "Invalid image format"
            )
        try:
            response = await self._route(image_np)
            return response
        except:
            raise HTTPException(status.HTTP_424_FAILED_DEPENDENCY, "Retrive failed")

    @app.get("/", status_code=status.HTTP_200_OK)
    async def home(self):
        return {"message": "Face Recogntion API"}
