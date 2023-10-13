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

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.apply_resize:
            input_batch = [
                np.array(Image.fromarray(image).resize((320, 240)))
                for image in batch[self.input_key]
            ]

        # if image have a same dimension
        else:
            input_batch = [image for image in batch[self.input_key]]

        input_batch_np = np.array(input_batch)

        boxes, batch_indices = self._model(input_batch_np)

        boxes_by_batch = self._model.split_boxes_by_batch(boxes, batch_indices)

        return {
            "boxes": boxes_by_batch,
            "original_image": batch[self.input_key],
        }
