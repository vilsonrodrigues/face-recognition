from typing import Dict, List
import numpy as np
from processing.face_postp import FacePostProcessing

class BatchFacePostProcessing(FacePostProcessing):

    def __init__(
        self,
        input_key: str = 'image',
        output_key: str = 'faces'
    ):
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

    def __call__(
        self,
        batch: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:

        faces: List[np.ndarray] = []

        for img, boxes in zip(batch[self.input_key],
                              batch['boxes']):

            """ 1 object per image """

            img_shape = img.shape

            # if no face detected, add original image
            if len(boxes[0]) == 0:

                face = img

            else:

                box_rescaled = FacePostProcessing.rescale_boxes(
                    boxes=boxes,
                    height=img_shape[0],
                    width=img_shape[1]
                )

                # if many boxes detected, keep the first
                face = FacePostProcessing.crop_object(
                    img,
                    box_rescaled[0])

                faces.append(face)

        return {self.output_key: faces}