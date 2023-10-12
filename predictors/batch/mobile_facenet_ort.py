from typing import Dict, Optional
import numpy as np
from PIL import Image
from face_embedding.mobile_facenet_ort import MobileFaceNetORT

class MobileFaceNetORTBatchPredictor:

    def __init__(
        self,
        model_path: str,
        input_key: str = 'image',
        backend: Optional[str] = None,
    ):
        """
        Args:
            model_path: path to model
            input_key: key to acess batch in inference
            backend: openvino or cuda for fast inference
        """
        self.backend = backend
        self.input_key = input_key
        self._model = MobileFaceNetORT(model_path, backend)

    def __call__(
        self,
        batch: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:

        # the model accepts any shape, but for concantenate image batch
        # is necessary same dimensions, so, resize is used

        input_batch = [np.array(Image.fromarray(image).resize((112, 112))) for image in batch[self.input_key]]

        input_batch_np = np.array(input_batch)

        embeddings = self._model(input_batch_np)[0]

        embeddings = embeddings.tolist()

        return {
            'embedding': embeddings,
            'face': batch[self.input_key],
        }