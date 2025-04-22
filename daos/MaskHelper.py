"""

- maskSubtitleBBoxes
- (image, mask)

- maskSubtitle
- (image)
- applyStructureGuidance
- (frame, mask)
- maskOverlay
- (image, mask)
- refine_bbox
- (image, easyocr_results)


"""


import time
from daos.TextDetector import TextDetector
from utils.logger import log_function, setup_logger
import numpy as np
import cv2
import easyocr
import os
import matplotlib.pyplot as plt
from daos.ImageHelper import ImageHelper


class MaskHelper:
  logger = setup_logger('MaskHelper')

  def __init__(self):
    self.textDetector = TextDetector(model="easyocr")
    self.imageHelper = ImageHelper()

  # Overlay the mask on the image to have a better visualization
  def maskOverlay(image: np.ndarray, mask: np.ndarray, hex_color: str = "#FF0000") -> np.ndarray:
    """
    Overlay mask on image with a custom HEX color.

    Args:
        image (np.ndarray): Original image (BGR).
        mask (np.ndarray): Binary mask.
        hex_color (str): HEX color string, e.g., "#00FF00" for green.

    Returns:
        np.ndarray: Image with mask overlay.
    """
    # Convert HEX to BGR
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    bgr = tuple(reversed(rgb))

    # Create colored mask
    color_mask = np.zeros_like(image)
    for i in range(3):  # B, G, R
      color_mask[:, :, i] = (mask > 0) * bgr[i]

    # Blend original image with mask
    return cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

  def refine_bbox(self, image: np.ndarray, easyocr_results) -> list:
    """Refine bounding boxes to better match character shapes using contour detection."""
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    refined_boxes = []
    for result in easyocr_results:
      bbox = result[0]
      # Extract region around the bbox
      x_min = min(point[0] for point in bbox)
      x_max = max(point[0] for point in bbox)
      y_min = min(point[1] for point in bbox)
      y_max = max(point[1] for point in bbox)

      # Get region of interest
      roi = thresh[int(y_min):int(y_max), int(x_min):int(x_max)]

      # Find contours in the ROI
      contours = cv2.findContours(roi, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)[0]

      # Process each contour as potential character
      for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Add offset to get coordinates in original image
        refined_boxes.append([
            x + x_min, y + y_min,
            x + x_min + w, y + y_min + h
        ])

    return refined_boxes

  @log_function(logger)
  def maskSubtitle(self, image: np.ndarray) -> np.ndarray:
    """Detects text in the image and returns a binary mask of detected text regions."""
    reader = easyocr.Reader(['en'], gpu=True)
    results = reader.readtext(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Get refined character-level bounding boxes
    refined_boxes = self.refine_bbox(image, results)

    # Create mask from refined boxes
    for box in refined_boxes:
      x1, y1, x2, y2 = box
      cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

    # # Visualize the process
    # self.visualize_mask_creation(image, results, refined_boxes, mask)

    return mask

  # Return the mask directly produced from BBoxes
  def maskSubtitleBBoxes(self, image: np.ndarray) -> np.ndarray:
    """Detects text in the image and returns a binary mask of detected text regions."""

    bboxes = self.textDetector.detect(
        self.imageHelper.applyCLAHE(image, mode='color'))
    # bboxes = self.textDetector.detect(image)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for (bbox) in bboxes:
      pts = np.array(bbox, dtype=np.int32)
      cv2.fillPoly(mask, [pts], 255)

    return mask

  def applyStructureGuidance(self, frame: np.ndarray, mask: np.ndarray, ksize: int = 7) -> np.ndarray:
    # start = time.time()

    # To make edges more visible
    frame = self.imageHelper.applyCLAHE(frame, mode='color')

    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200)
    # print(f"Canny time: {time.time() - start}")
    # start = time.time()
    # edges = cv2.Laplacian(cv2.cvtColor(
    #     frame, cv2.COLOR_BGR2GRAY), cv2.CV_8U, ksize=3)
    # print(f"Laplacian time: {time.time() - start}")
    boost_mask = cv2.dilate(edges, np.ones(
        (ksize, ksize), np.uint8), iterations=1)
    boost_mask = cv2.bitwise_and(boost_mask, mask)

    cv2.imwrite("boost_mask.png", boost_mask)

    return boost_mask
