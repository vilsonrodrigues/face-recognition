from typing import Generator, List

import numpy as np
from ray import serve

from processing.face_postp import FacePostProcessing
from processing.utils import check_img_channels

@serve.deployment(
    name="FacePostProcessing",
    ray_actor_options={
        "num_gpus": 0.0,
        "num_cpus": 0.05,
        "memory": 100,
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

    async def apply_rescale_boxes(
        self, relative_boxes: np.ndarray, img: np.ndarray
    ) -> List[List[int]]:
        """Apply rescale in boxes
        Args:
            relative_boxes: generated boxes
            img: original image
        Returns:
            a boxes rescaled list
        """
        img_shape = img.shape
        boxes_rescaled = FacePostProcessing.rescale_boxes(
            boxes=relative_boxes, height=img_shape[0], width=img_shape[1]
        )
        return boxes_rescaled

    async def apply_crop_faces(
        self, boxes_rescaled: List[List[int]], img: np.ndarray
    ) -> Generator[np.ndarray, None, None]:
        """Crop faces
        Args:
            boxes_rescaled: a boxes rescaled list
            img: original image
        Returns:
            a generator of cropped face
        """
        for box_rescaled in boxes_rescaled:
            face = FacePostProcessing.crop_object(img, box_rescaled)
            # checks if any image dimension is invalid, if so, keep the original image
            if check_img_channels(face) == False:
                raise ValueError("Cut face has invalid dimensions")
            else:
                yield face      
