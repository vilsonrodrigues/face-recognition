from typing import Optional, List
import numpy as np
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

        The output is `(BOXES, 4)` and `(BATCH_INDICES)`

    The BATCH_INDICES will tell you which batch the boxes belong to

    Default thresholds values are:
        IoU: 0.5
        Score: 0.95

    Backend avaliable: `openvino` (CPU_FP16) or `cuda`
    """

    def __init__(self, model_path: str, backend: Optional[str] = None):
        super().__init__(model_path, backend)

    def split_boxes_by_batch(
        self, boxes: np.ndarray, batch_indices: np.ndarray, max_index: int
    ) -> List[np.ndarray]:
        """Split boxes by batch
        Args:
            boxes: (BOXES, 4)
            batch_indices: (BATCH_INDICES) tells which batch the boxes belong to
            max_index: max batch size
        Returns:
            boxes_by_batch
        """

        boxes_by_batch = []

        for batch_idx in range(max_index):
            if batch_idx in batch_indices:
                batch_boxes = boxes[batch_indices == batch_idx]
                # fix negative values
                batch_boxes = np.abs(batch_boxes)                

            # if not boxes detected, add a empty numpy array
            else:
                # empty array with 2-dim 
                batch_boxes = np.array([[],[]], dtype=np.float32)

            boxes_by_batch.append(batch_boxes)

        return boxes_by_batch
