from typing import Optional
from base.ort_model import ONNXRuntimeModel

class MobileFaceNetORT(ONNXRuntimeModel):

    """
    This model is a lightweight face embedding model
    designed for edge computing devices.

    Input tensor is `(N x 3 x 112 x 112)` with mean
    values `(127, 127, 127)` and scale factor `1.0 / 128.`

    The model outputs is an embedding array `(N x 128)`

    The model `mobilefacenet_prep.onnx` contains
    preprocessing layers. The input is `(N x H x W x 3)`

    """    

    def __init__(self, model_path: str, backend: Optional[str] = None):
        super().__init__(model_path, backend)