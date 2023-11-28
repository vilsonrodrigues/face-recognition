from typing import Dict, Optional
import numpy as np
from PIL import Image
from face_detection.ultra_light_ort import UltraLightORT


class UltraLightORTBatchPredictor:
    def __init__(
        self,
        model_path: str,
        input_key: str = "image",
        apply_resize: Optional[bool] = False,
        backend: Optional[str] = None,
    ):
        """
        Args:
            model_path: path to model
            input_key: key to acess batch in inference
            apply_resize: if apply resize to image inputs
            backend: openvino or cuda for fast inference
        """

        self.apply_resize = apply_resize
        self.backend = backend
        self.input_key = input_key
        self._model = UltraLightORT(model_path, backend)

    def __call__(self, input_batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # if image input not have a same dimensions
        if self.apply_resize:
            batch = [
                np.expand_dims(
                    np.array(
                        Image.fromarray(image).resize(
                            (320, 240), Image.Resampling.LANCZOS
                        )
                    ),
                    axis=0,
                )
                for image in input_batch[self.input_key]
            ]

        else:
            batch = [
                np.expand_dims(image, axis=0) for image in input_batch[self.input_key]
            ]

        concatenated_batch = np.concatenate(batch, axis=0)

        boxes, batch_indices = self._model(concatenated_batch)

        max_batch_size = len(batch)

        # if no face is detected in any batch
        if len(batch_indices) == 0:
            # add empty arrays for each batch
            boxes_by_batch = [np.array([[], []]) for _ in range(len(max_batch_size))]

        else:
            boxes_by_batch = self._model.split_boxes_by_batch(
                boxes, batch_indices, max_batch_size
            )

        boxes_by_batch = self._model.split_boxes_by_batch(boxes, batch_indices)

        return {
            "boxes": boxes_by_batch,
            "original_image": input_batch[self.input_key],
        }
