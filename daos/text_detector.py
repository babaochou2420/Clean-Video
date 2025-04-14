import cv2
import numpy as np
import easyocr
from utils.logger import setup_logger, log_function

logger = setup_logger('text_detector')


class TextDetector:
  def __init__(self, lang_list=None):
    logger.info("Initializing TextDetector")
    self.reader = easyocr.Reader(lang_list or ['en'], gpu=True)

  @log_function(logger)
  def create_subtitle_mask(self, image: np.ndarray) -> np.ndarray:
    """Detects text in the image and returns a binary mask of detected text regions."""
    results = self.reader.readtext(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for (bbox, text, confidence) in results:
      pts = np.array(bbox, dtype=np.int32)
      cv2.fillPoly(mask, [pts], 255)

    return mask

  @log_function(logger)
  def visualize_detection(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay mask on image as red area for visualization."""
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = mask  # Red channel
    vis = cv2.addWeighted(image, 0.7, red_mask, 0.3, 0)
    return vis
