from base.ort_model import ONNXRuntimeModel

class MobileFaceNet(ONNXRuntimeModel):

    """
    This model is a lightweight face embedding model
    designed for edge computing devices.

    Input tensor is `(N x 3 x 112 x 112)` with mean
    values `(127, 127, 127)` and scale factor `1.0 / 128.`

    The model outputs is an embedding array `(N x 128)`
    """

    def __init__(self, MODEL_PATH: str) -> None:
        super().__init__(MODEL_PATH)

    def get_embedding_dim(self) -> int:
        return self._model.get_outputs()[0].shape[1]