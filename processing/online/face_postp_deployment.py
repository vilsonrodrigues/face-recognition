from typing import Dict, List, Union

import numpy as np
from ray import serve

from processing.face_postp import FacePostProcessing


@serve.deployment(
    name="FacePostProcessing",
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
class FacePostProcessingDeployment(FacePostProcessing):
    def __init__(self):
        super().__init__()

    async def apply(
        self, relative_boxes: np.ndarray, img: np.ndarray
    ) -> Dict[str, List[Union[np.ndarray, int]]]:
        img_shape = img.shape

        # if no face detected, return a empty list
        if len(relative_boxes) == 0:
            return {
                "faces": [],
                "boxes_rescaled": [],
            }

        else:
            faces: List[np.ndarray] = []

            boxes_rescaled = FacePostProcessing.rescale_boxes(
                boxes=relative_boxes, height=img_shape[0], width=img_shape[1]
            )

            for box_rescaled in boxes_rescaled:
                face = FacePostProcessing.crop_object(img, box_rescaled)

                faces.append(face)

            return {
                "faces": faces,
                "boxes_rescaled": boxes_rescaled,
            }
