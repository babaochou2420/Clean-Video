import cv2
import numpy as np


class ImageHelper:
  def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
    self.clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=tile_grid_size)

  def applyCLAHE(self, image: np.ndarray, mode='gray') -> np.ndarray:
    """
    Apply CLAHE to the input image.
    Args:
        image (np.ndarray): Input BGR image.
        mode (str): 'gray' to apply CLAHE on grayscale,
                    'color' to apply CLAHE on each BGR channel separately.
    Returns:
        np.ndarray: Image with CLAHE applied.
    """
    if mode == 'gray':
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      clahe_applied = self.clahe.apply(gray)
      result = cv2.cvtColor(clahe_applied, cv2.COLOR_GRAY2BGR)

    elif mode == 'color':
      channels = cv2.split(image)
      clahe_channels = [self.clahe.apply(ch) for ch in channels]
      result = cv2.merge(clahe_channels)

    else:
      raise ValueError("Mode must be either 'gray' or 'color'.")

    cv2.imwrite(
        "clahe_applied.png", result)
    return result
