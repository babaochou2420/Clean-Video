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

  @log_function(logger)
  def maskOverlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay mask on image as red area for visualization."""
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = mask  # Red channel
    vis = cv2.addWeighted(image, 0.7, red_mask, 0.3, 0)
    return vis

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

  def visualize_mask_creation(self, image: np.ndarray, results, refined_boxes, final_mask: np.ndarray):
    """Visualize the step-by-step process of mask creation and save to PNG."""
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Original image with EasyOCR boxes
    img_with_boxes = image.copy()
    for bbox in results:
      pts = np.array(bbox[0], dtype=np.int32)
      cv2.polylines(img_with_boxes, [pts], True, (0, 255, 0), 2)
    axes[0, 0].imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image with EasyOCR Boxes')
    axes[0, 0].axis('off')

    # Grayscale and thresholded image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    axes[0, 1].imshow(thresh, cmap='gray')
    axes[0, 1].set_title('Thresholded Image')
    axes[0, 1].axis('off')

    # Image with refined character boxes
    img_with_refined = image.copy()
    for box in refined_boxes:
      x1, y1, x2, y2 = box
      cv2.rectangle(img_with_refined, (int(x1), int(y1)),
                    (int(x2), int(y2)), (0, 0, 255), 2)
    axes[1, 0].imshow(cv2.cvtColor(img_with_refined, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Image with Refined Character Boxes')
    axes[1, 0].axis('off')

    # Final mask
    axes[1, 1].imshow(final_mask, cmap='gray')
    axes[1, 1].set_title('Final Mask')
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Save the figure to a PNG file
    output_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'tests', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'maskCreation_v2.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

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
        self.imageHelper.applyCLAHE(image, mode='gray'))

    # bboxes = self.textDetector.detect(image)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for (bbox) in bboxes:
      pts = np.array(bbox, dtype=np.int32)
      cv2.fillPoly(mask, [pts], 255)

    return mask

  def applyStructureGuidance(self, frame: np.ndarray, mask: np.ndarray, ksize: int = 7) -> np.ndarray:
    # start = time.time()

    # Apply CLAHE to the frame
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
