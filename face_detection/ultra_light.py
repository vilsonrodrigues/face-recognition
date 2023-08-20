from base.ort_model import ONNXRuntimeModel

class UltraLight(ONNXRuntimeModel):

    """
    This model is a lightweight face detection model
    designed for edge computing devices.

    Input tensor is `(1 x 3 x height x width)` with mean
    values `(127, 127, 127)` and scale factor `1.0 / 128.`

    Input image have to be previously converted to
    RGB format and resized to `320 x 240` pixels for
    version-RFB-320 model
    (or 640 x 480 for version-RFB-640 model).

    The model outputs two arrays `(2 x 4420 x 2)` and
    `(4 x 4420 x 4)` of scores and boxes.
    """

    def __init__(self, MODEL_PATH: str):
        super().__init__(MODEL_PATH)