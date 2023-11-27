from typing import Dict, Optional
import numpy as np
from PIL import Image
from face_embedding.mobile_facenet_ort import MobileFaceNetORT


class MobileFaceNetORTBatchPredictor:
    def __init__(
        self,
        model_path: str,
        input_key: str = "image",
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

    def __call__(self, input_batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # the model accepts any shape, but for concantenate image batch
        # is necessary same dimensions, so, resize is used

        batch = [
            np.expand_dims(
                np.array(Image.fromarray(image).resize((112, 112), Image.ANTIALIAS)),
                axis=0,
            )
            for image in input_batch[self.input_key]
        ]

        concatenated_batch = np.concatenate(batch, axis=0)

        embeddings = self._model(concatenated_batch)[0]

        embeddings = embeddings.tolist()

        return {
            "embedding": embeddings,
            "face": input_batch[self.input_key],
        }
