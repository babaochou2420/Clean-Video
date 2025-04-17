import cv2
import numpy as np
import easyocr
from utils.logger import setup_logger, log_function

logger = setup_logger('text_detector')


class TextDetector:
  def __init__(self, lang_list=None):
    logger.info("Initializing TextDetector")
    self.reader = (easyocr.Reader(lang_list or ['en'], gpu=True), "easyocr")

  @log_function(logger)
  def visualize_detection(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay mask on image as red area for visualization."""
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = mask  # Red channel
    vis = cv2.addWeighted(image, 0.7, red_mask, 0.3, 0)
    return vis

  def detect(self, image: np.ndarray):
    match self.reader[1]:
      case "easyocr":
        results = self.reader[0].readtext(image)
      case _:
        raise ValueError(f"Unsupported reader: {self.reader[1]}")
    return results
