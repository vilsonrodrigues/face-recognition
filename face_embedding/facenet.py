from base.ort_model import ONNXRuntimeModel

class FaceNet(ONNXRuntimeModel):

    def __init__(self, MODEL_PATH: str) -> None:
        super().__init__(MODEL_PATH)

    def get_embedding_dim(self) -> int:
        return self._model.get_outputs()[0].shape[1]