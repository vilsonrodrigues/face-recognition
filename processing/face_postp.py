from typing import List
import numpy as np


class FacePostProcessing:
    def __init__(self):
        pass

    @staticmethod
    def crop_object(img: np.ndarray, box: List[int]) -> np.ndarray:
        return img[box[1] : box[3], box[0] : box[2]]

    @staticmethod
    def convert_boxes_relative_to_absolute(
        boxes: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        boxes_copy = boxes.copy()
        boxes_copy[:, 0] *= width
        boxes_copy[:, 1] *= height
        boxes_copy[:, 2] *= width
        boxes_copy[:, 3] *= height
        return boxes_copy

    @staticmethod
    def rescale_box(box: np.ndarray) -> List[int]:
        width = box[2] - box[0]
        height = box[3] - box[1]
        maximum = max(width, height)
        dx = int((maximum - width) / 2)
        dy = int((maximum - height) / 2)
        bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
        return bboxes

    @staticmethod
    def rescale_boxes(boxes: np.ndarray, height: int, width: int) -> List[List[int]]:
        """
        Args:
            width: original image width
            height: original image height
            boxes: (N, 4) boxes array

        Returns:
            boxes_list: a list with reescaled boxes
        """
        boxes_absolute = FacePostProcessing.convert_boxes_relative_to_absolute(
            boxes, height, width
        )
        boxes_absolute_int = boxes_absolute.astype(np.int32)
        boxes_list: List[np.ndarray] = []
        for i in range(boxes_absolute_int.shape[0]):
            box = FacePostProcessing.rescale_box(boxes_absolute_int[i, :])
            boxes_list.append(box)
        return boxes_list
