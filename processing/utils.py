import io
import base64
import numpy as np
from PIL import Image

def check_img_channels(img: np.ndarray) -> np.ndarray:
    """This function verify image dimension to
    to ensure you have correct color channel
    Args:
        img

    Returns:
        frame: Frame with corrects channels
    """
    #If image 2D, add 1D to switch to 3D
    if len(img.shape) == 2:
        return np.reshape(img, img.shape + (1,))
    else:
        return img

def convert_bytes_to_base64(img_bytes: bytes) -> str:
    encoded_image = base64.b64encode(img_bytes).decode('utf-8')
    return encoded_image

def convert_bytes_to_numpy(img_bytes: bytes) -> np.ndarray:
    img_decode = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img_decode)
    return img_array

def l2_norm_on_tensor(tensor: np.ndarray) -> np.ndarray:
    img_l2_norm = tensor / np.linalg.norm(tensor, axis=1, keepdims=True)
    return img_l2_norm        