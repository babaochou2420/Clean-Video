import cv2
import numpy as np
import easyocr
from utils.logger import setup_logger, log_function

from paddleocr import PaddleOCR

logger = setup_logger('text_detector')


class TextDetector:
  def __init__(self, lang_list=None, model: str = "easyocr"):
    logger.info("Initializing TextDetector")

    match model:
      case "easyocr":
        self.reader = (easyocr.Reader(
            lang_list or ['en'], gpu=True), "easyocr")
      case "paddle":
        # PaddleOCR supports multiple languages, default is 'ch' (Chinese)
        # You can specify multiple languages like: lang='chinese_english'
        # or use specific language codes like 'en', 'fr', 'de', etc.
        self.reader = (
            PaddleOCR(lang=[] if lang_list else 'ch'), "paddle")

  def visualize_detection(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay mask on image as red area for visualization."""
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = mask  # Red channel
    vis = cv2.addWeighted(image, 0.7, red_mask, 0.3, 0)
    return vis

  def detect(self, image: np.ndarray):
    match self.reader[1]:
      case "easyocr":
        # o: [(bbox, text, confidence)]
        results = self.reader[0].readtext(image)

        # logger.info(results)

        bboxes = [item[0] for item in results]
      case "paddle":
        # o: [bbox]
        results = self.reader[0].ocr(image, det=False, cls=False, rec=False)
        # logger.info(results)
        bboxes = results
      case _:
        raise ValueError(f"Unsupported reader: {self.reader[1]}")
    return bboxes
