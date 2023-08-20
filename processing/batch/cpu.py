from typing import Dict, List
import numpy as np
from processing.utils import (check_img_channels,
                              convert_bytes_to_base64,
                              convert_bytes_to_numpy,
                              crop_object,
                              normalize_img,
                              resize_img,
                              transpose_channels
                              )
from processing.utils_face_detection import apply_nms, rescale_bbox

def batch_convert_bytes_to_base64(
    batch: Dict[str, bytes],
    input_key: str = 'foto',
    output_key: str = 'base64'
) -> Dict[str, bytes]:
    images: List[np.ndarray] = []
    for img_bytes in batch[input_key]:
        image = convert_bytes_to_base64(img_bytes)
        images.append(image)
    batch[output_key] = images
    return batch

def batch_convert_bytes_to_numpy(
    batch: Dict[str, bytes],
    input_key: str = 'foto',
    output_key: str = 'image'
) -> Dict[str, np.ndarray]:
    images: List[np.ndarray] = []
    for img_bytes in batch[input_key]:
        image = convert_bytes_to_numpy(img_bytes)
        images.append(image)
    batch[output_key] = np.array(images)
    return batch

def batch_check_img_channels(
    batch: Dict[str, np.ndarray],
    input_key: str = 'image',
    output_key: str = 'image'
) -> Dict[str, np.ndarray]:
    images: List[np.ndarray] = []
    for img in batch[input_key]:
        image = check_img_channels(img)
        images.append(image)
    batch[output_key] = np.array(images)
    return batch

def batch_resize_img(
    batch: Dict[str, np.ndarray],
    width: int = 160,
    height: int = 160,
    input_key: str = 'image',
    output_key: str = 'image_resized'
) -> Dict[str, np.ndarray]:
    images: List[np.ndarray] = []
    for img in batch[input_key]:
        image = resize_img(img, width, height)
        images.append(image)
    batch[output_key] = np.array(images)
    return batch

def batch_normalize_img(
    batch: Dict[str, np.ndarray],
    input_key: str = 'image_resized',
    output_key: str = 'image_normalized'
) -> Dict[str, np.ndarray]:
    images: List[np.ndarray] = []
    for image_resized in batch[input_key]:
        image = normalize_img(image_resized)
        images.append(image)
    batch[output_key] = np.array(images)
    return batch

def batch_transpose_channels(
    batch: Dict[str, np.ndarray],
    input_key: str = 'image_normalized',
    output_key: str = 'image_transposed'
) -> Dict[str, np.ndarray]:
    images: List[np.ndarray] = []
    for image_normalized in batch[input_key]:
        image = transpose_channels(image_normalized)
        images.append(image)
    batch[output_key] = np.array(images)
    return batch

# post processing ultralight
def batch_apply_nms(
    batch: Dict[str, np.ndarray],
    prob_threshold: float = 0.95,
    iou_threshold: float = 0.5,
    input_key: str = 'image',
    output_key: str = 'bboxs'
) -> Dict[str, np.ndarray]:
    bboxs: List[np.ndarray] = []
    for image, confidences, boxes in zip(batch[input_key],
                                         batch['confidences'],
                                         batch['boxes']):

        bbox, _, _ = apply_nms(image.shape[1],
                               image.shape[0],
                               confidences,
                               boxes,
                               prob_threshold=prob_threshold,
                               iou_threshold=iou_threshold)

        bboxs.append(bbox)
    batch[output_key] = np.array(bboxs)
    return batch

def batch_crop_face(
    batch: Dict[str, np.ndarray],
    input_key: str = 'image',
    output_key: str = 'face'
) -> Dict[str, np.ndarray]:
    """ 1 object per image """
    faces: List[np.ndarray] = []
    for image, bbox in zip(batch[input_key], batch['bboxs']):

        # if no face detected, add original image
        if len(bbox[0]) == 0:
            face = image

        else:
            box_rescaled = rescale_bbox(bbox[0])
            face = crop_object(image, box_rescaled)

        faces.append(face)
    batch[output_key] = np.array(faces)
    return batch