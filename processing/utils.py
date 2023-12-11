import io
import base64

import numpy as np
from PIL import Image


def check_img_channels(image: np.ndarray) -> bool:
    """
    This function checks the channels of a given image.

    The function performs the following checks:
    0. Checks if there are three color channels.
    1. Checks if the first channel is greater than 0.
    2. Checks if the second channel is greater than 0.
    3. Checks if the third channel is greater than 0.

    If any of these checks fail, the function returns False. Otherwise, it returns True.

    Args:
        image: The input image in numpy format.

    Returns:
        True if all checks pass, False otherwise.
    """
    # Check if there are three color channels
    if image.ndim != 3 or image.shape[2] != 3:
        return False

    # Check if the first channel is greater than 0
    if np.max(image[:, :, 0]) <= 0:
        return False

    # Check if the second channel is greater than 0
    if np.max(image[:, :, 1]) <= 0:
        return False

    # Check if the third channel is greater than 0
    if np.max(image[:, :, 2]) <= 0:
        return False

    return True


def convert_bytes_to_base64(img_bytes: bytes) -> str:
    encoded_image = base64.b64encode(img_bytes).decode("utf-8")
    return encoded_image


def convert_bytes_to_numpy(img_bytes: bytes) -> np.ndarray:
    img_decode = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img_decode)
    return img_array


def l2_norm_on_tensor(tensor: np.ndarray) -> np.ndarray:
    img_l2_norm = tensor / np.linalg.norm(tensor, axis=1, keepdims=True)
    return img_l2_norm


def convert_numpy_to_base64(array: np.ndarray) -> bytes:
    array_bytes = array.tobytes()
    base64_encoded = base64.b64encode(array_bytes)
    base64_string = base64_encoded.decode("utf-8")
    return base64_string
