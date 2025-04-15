
from utils.logger import log_function, setup_logger
import numpy as np
import cv2


class MaskHelper:
  logger = setup_logger('MaskHelper')

  def __init__(self):
    pass

  @log_function(logger)
  def maskOverlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay mask on image as red area for visualization."""
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = mask  # Red channel
    vis = cv2.addWeighted(image, 0.7, red_mask, 0.3, 0)
    return vis
