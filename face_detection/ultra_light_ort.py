from typing import Optional
from base.ort_model import ONNXRuntimeModel

class UltraLightORT(ONNXRuntimeModel):

    """
    This model is a lightweight face detection model
    designed for edge computing devices.

    Input tensor is `(N x 3 x height x width)` with mean
    values `(127, 127, 127)` and scale factor `1.0 / 128.`

    Input image have to be previously converted to
    RGB format and resized to `320 x 240` pixels for
    version-RFB-320 model
    (or 640 x 480 for version-RFB-640 model).

    The model outputs two arrays `(N x 4420 x 2)` and
    `(N x 4420 x 4)` of scores and boxes.

    The model version `ultralight_RBF_320_prep_nms.onnx`
    contains preprocessing and NMS layers.

        The input is `(N x H x W x 3)`

        The output is `(N, BOXES)`

    Default thresholds values are:
        IoU: 0.5
        Score: 0.95

    Backend avaliable: `openvino` (CPU_FP16) or `cuda`
    """

    def __init__(self, model_path: str,  backend: Optional[str] = None):
        super().__init__(model_path, backend)