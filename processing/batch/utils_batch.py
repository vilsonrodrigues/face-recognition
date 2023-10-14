from typing import Dict, List
import numpy as np
from processing.utils import (
    check_img_channels,
    convert_bytes_to_base64,
    convert_bytes_to_numpy,
    convert_numpy_to_base64,
)


def batch_convert_bytes_to_base64(
    batch: Dict[str, bytes], 
    input_key: str = "foto", 
    output_key: str = "base64"
) -> Dict[str, bytes]:
    images: List[np.ndarray] = []
    for img_bytes in batch[input_key]:
        image = convert_bytes_to_base64(img_bytes)
        images.append(image)
    batch[output_key] = images
    return batch


def batch_convert_bytes_to_numpy(
    batch: Dict[str, bytes], 
    input_key: str = "foto", 
    output_key: str = "image"
) -> Dict[str, np.ndarray]:
    images: List[np.ndarray] = []
    for img_bytes in batch[input_key]:
        image = convert_bytes_to_numpy(img_bytes)
        images.append(image)
    batch[output_key] = images
    return batch


def batch_check_img_channels(
    batch: Dict[str, np.ndarray], 
    input_key: str = "image", 
    output_key: str = "image"
) -> Dict[str, np.ndarray]:
    images: List[np.ndarray] = []
    for img in batch[input_key]:
        image = check_img_channels(img)
        images.append(image)
    batch[output_key] = images
    return batch


def batch_convert_numpy_to_base64(
    batch: Dict[str, np.ndarray],
    input_key: str = "face",
    output_key: str = "face_base64",
) -> Dict[str, np.ndarray]:
    images: List[np.ndarray] = []
    for img in batch[input_key]:
        image = convert_numpy_to_base64(img)
        images.append(image)
    batch[output_key] = images
    return batch
