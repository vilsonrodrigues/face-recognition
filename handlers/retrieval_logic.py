import asyncio
from typing import Any, Dict, List, Union

import numpy as np
from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse


@serve.deployment(
    name="RetrievalHandler",
    ray_actor_options={
        "num_gpus": 0.0,
        "num_cpus": 0.2,
        "memory": 600,
        "runtime_env": {
            "image": "rayproject/ray:2.7.1-py310",
            # "run_options": ["--cap-drop SYS_ADMIN","--log-level=debug"]
        },
    },
    health_check_period_s=10,
    health_check_timeout_s=30,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 2,
        "initial_replicas": 1,
    },
)
class PipelineRetrievalLogic:
    def __init__(
        self,
        face_emb: DeploymentHandle,
        face_postp: DeploymentHandle,
        neural_search: DeploymentHandle,
    ):
        self._face_emb: DeploymentHandle = face_emb.options(
            use_new_handle_api=True,
        )
        self._face_postp: DeploymentHandle = face_postp.options(
            use_new_handle_api=True,
        )
        self._neural_search: DeploymentHandle = neural_search.options(
            use_new_handle_api=True,
        )

    async def route(
        self,
        relative_boxes: List[np.ndarray],
        image: np.ndarray,
    ) -> Dict[str, Union[str, List[Dict[str, Any]]]]:
        # if no face detected, return a empty list
        if len(relative_boxes) == 0:
            return {
                "payloads": [],
                "boxes": [],
            }

        else:
            # async call to face postp deployment
            faces_and_boxes_rescaled_ref: DeploymentResponse = (
                self._face_postp.apply.remote(relative_boxes, image)
            )

            # await is required to iterate over detected faces
            faces_and_boxes_rescaled = await faces_and_boxes_rescaled_ref

            # async call to face emb and neural search deployment
            tasks = [
                self._neural_search.search.remote(self._face_emb.predict.remote(face))
                for face in faces_and_boxes_rescaled["faces"]
            ]

            # gather payloads
            payloads = await asyncio.gather(*tasks)

            return {
                "payloads": payloads,
                "boxes": faces_and_boxes_rescaled["boxes_rescaled"],
            }
