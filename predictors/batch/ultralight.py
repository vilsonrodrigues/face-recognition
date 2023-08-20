from typing import Dict, List, Tuple
import numpy as np
from face_detection.ultra_light import UltraLight

class UltraLightBatchPredictor:

    def __init__(self, MODEL_PATH, INPUT_KEY: str = 'image'):
        self.INPUT_KEY = INPUT_KEY
        self._model = UltraLight(MODEL_PATH)

    def __call__(
        self,
        batch: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        input_batch = [image for image in batch[self.INPUT_KEY]]

        boxes_list = []
        confidences_list = []

        for face in input_batch:
            confidences, boxes = self._model(face)

            boxes_np = boxes.numpy()
            confidences_np = confidences.numpy()

            boxes_list.append(boxes_np)
            confidences_list.append(confidences_np)

        batch['boxes'] = np.array(boxes_list)
        batch['confidences'] = np.array(confidences_list)

        return batch