import io
from typing import List
import cv2
import base64
import numpy as np
from PIL import Image

def normalize_img(img: np.ndarray) -> np.ndarray:
    mean, std = img.mean(), img.std()
    img_norm = (img - mean) / std
    return img_norm

def resize_img(
    img: np.ndarray,
    width: int = 160,
    height: int = 160
) -> np.ndarray:
    img_resized = cv2.resize(img, (width, height))
    return img_resized

def l2_norm_on_tensor(tensor: np.ndarray) -> np.ndarray:
    img_l2_norm = tensor / np.linalg.norm(tensor, axis=1, keepdims=True)
    return img_l2_norm

def convert_bytes_to_numpy(img_bytes: bytes) -> np.ndarray:
    img_decode = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img_decode)
    return img_array

def convert_bytes_to_base64(img_bytes: bytes) -> str:
    encoded_image = base64.b64encode(img_bytes).decode('utf-8')
    return encoded_image

def check_img_channels(img: np.ndarray) -> np.ndarray:
    """This function verify image dimension to
    to ensure you have correct color channel
    Args:
        frameimg

    Returns:
        frame: Frame with corrects channels
    """
    #If image 2D, add 1D to switch to 3D
    if len(img.shape) == 2:
        return np.reshape(img, img.shape + (1,))
    else:
        return img

def transpose_channels(img: np.ndarray) -> np.ndarray:
    """ Adapt img for Pytorch format
        (W, H, C) -> (C, H, W)
    """
    img_transposed = np.transpose(img, [2, 0, 1])
    return img_transposed

def crop_object(img: np.ndarray, box: List[int]) -> np.ndarray:
    return img[box[1]:box[3], box[0]:box[2]]