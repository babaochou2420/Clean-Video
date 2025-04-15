from utils.logger import log_function, setup_logger
import numpy as np
import cv2
import easyocr
import os


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

  @log_function(logger)
  def maskSubtitle(self, image: np.ndarray) -> np.ndarray:
    """Detects text in the image and returns a binary mask of detected text regions."""
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for (bbox, text, confidence) in results:
      pts = np.array(bbox, dtype=np.int32)
      cv2.fillPoly(mask, [pts], 255)

    mask = self.addLineStructure(image, mask)

    return mask

  # def addLineStructure(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
  #   edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200)

  #   # Create absolute path for the edges image
  #   edges_path = os.path.join(os.path.dirname(
  #       os.path.dirname(__file__)), 'tests', 'edges.png')
  #   cv2.imwrite(edges_path, edges)

  #   # Dilate edges to connect nearby components
  #   kernel = np.ones((3, 3), np.uint8)
  #   edges = cv2.dilate(edges, kernel, iterations=2)

  #   # Only keep edges near the masked region
  #   dilated_mask = cv2.dilate(mask, kernel, iterations=5)

  #   # Combine original mask with nearby edges
  #   guided_mask = cv2.bitwise_or(mask, cv2.bitwise_and(edges, dilated_mask))

  #   # Clean up the final mask
  #   guided_mask = cv2.morphologyEx(guided_mask, cv2.MORPH_CLOSE, kernel)

  #   return guided_mask
