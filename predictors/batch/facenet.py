from typing import Any, Dict
import numpy as np
from face_embedding.facenet import FaceNet

class FaceNetBatchPredictor:

    def __init__(self, MODEL_PATH, INPUT_KEY: str = 'image'):
        """
        Args:
            MODEL_PATH: path to model
            INPUT_KEY: key to acess batch in inference
        """
        self.INPUT_KEY = INPUT_KEY
        self._model = FaceNet(MODEL_PATH)

    def __call__(
        self,
        batch: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        input_batch = [image for image in batch[self.INPUT_KEY]]

        input_batch_np = np.array(input_batch)

        embeddings = self._model(input_batch_np)[0].numpy()

        batch['embedding'] = embeddings

        return batch