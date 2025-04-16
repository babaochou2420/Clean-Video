from utils.logger import log_function, setup_logger
import numpy as np
import cv2
import easyocr
import os
import matplotlib.pyplot as plt

import rembg

from PIL import Image


class MaskHelper:
  logger = setup_logger('MaskHelper')

  def __init__(self):
    pass

  # def inpaint_subtitles_with_rembg(image, text_boxes):
  #   """
  #   Inpaints subtitles using RemBG to create a precise mask.

  #   Args:
  #       image (np.ndarray): The input image (BGR).
  #       text_boxes (list): List of bounding boxes for the text.

  #   Returns:
  #       np.ndarray: The inpainted image.
  #   """

  #   # Initialize the final mask
  #   final_mask = np.zeros(image.shape[:2], dtype=np.uint8)

  #   for box in text_boxes:
  #     x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
  #     text_region = image[y1:y2, x1:x2]  # Extract the text region

  #     # Use RemBG to remove the background
  #     output = remove(text_region)  # RemBG processing
  #     # The output of rembg is an image with alpha channel.

  #     # Extract the alpha channel as the mask and threshold it.
  #     alpha_channel = output[:, :, 3]
  #     _, text_mask = cv2.threshold(alpha_channel, 128, 255, cv2.THRESH_BINARY)

  #     #  Place the text mask into the final mask at the correct location
  #     final_mask[y1:y2, x1:x2] = text_mask

  @log_function(logger)
  def maskOverlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay mask on image as red area for visualization."""
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = mask  # Red channel
    vis = cv2.addWeighted(image, 0.7, red_mask, 0.3, 0)
    return vis

  # @log_function(logger)
  # def maskSubtitle(self, image: np.ndarray) -> np.ndarray:
  #   """Detects text in the image and returns a binary mask of detected text regions."""
  #   reader = easyocr.Reader(['en'], gpu=True)
  #   results = reader.readtext(image)
  #   mask = np.zeros(image.shape[:2], dtype=np.uint8)

  #   for (bbox, text, confidence) in results:
  #     pts = np.array(bbox, dtype=np.int32)
  #     cv2.fillPoly(mask, [pts], 255)

  #   mask = self.addLineStructure(image, mask)

  #   return mask

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

    # # Get refined character-level bounding boxes
    # refined_boxes = self.refine_bbox(image, results)

    # # Create mask from refined boxes
    # for box in refined_boxes:
    #   x1, y1, x2, y2 = box
    #   cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

    # Add the precise mask to the overall mask
    # mask = cv2.bitwise_or(mask, region_mask)

    # # Visualize the process
    # self.visualize_mask_creation(image, results, refined_boxes, mask)

    for (bbox, text, confidence) in results:
      mask = self.generate_text_mask_with_rembg(image, bbox)
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

  def generate_text_mask_with_rembg(self, image: np.ndarray, bbox: list) -> np.ndarray:
    """
    For each bounding box, crop the region, apply rembg to extract text-like foreground,
    and merge all resulting masks into one binary mask.

    Args:
        image (np.ndarray): Input BGR image.
        bboxes (list): List of bounding boxes, each box in format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] (easyocr format)

    Returns:
        np.ndarray: Final binary mask with detected text regions (0 or 255).
    """
    height, width = image.shape[:2]
    final_mask = np.zeros((height, width), dtype=np.uint8)
    # Get bounding box coordinates
    x_coords = [int(pt[0]) for pt in bbox]
    y_coords = [int(pt[1]) for pt in bbox]
    x1, y1, x2, y2 = min(x_coords), min(
        y_coords), max(x_coords), max(y_coords)

    # Crop the region
    cropped = image[y1:y2, x1:x2]
    # Preprocessing
    # - Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cropped = clahe.apply(cropped)
    # - Sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    cropped = cv2.filter2D(cropped, -1, kernel)

    # Convert to PIL and apply rembg
    pil_cropped = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    try:
      rembg_output = rembg.remove(pil_cropped)
    except Exception as e:
      print(f"rembg failed on bbox {bbox}: {e}")

    # Convert back to mask (alpha channel is the mask)
    rembg_np = np.array(rembg_output)
    if rembg_np.shape[2] == 4:
      alpha = rembg_np[:, :, 3]
    else:
      alpha = np.all(rembg_np != [0, 0, 0], axis=2).astype(np.uint8) * 255

    # Resize alpha mask to original bbox size and place into final mask
    mask_region = cv2.resize(
        alpha, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    final_mask[y1:y2, x1:x2] = cv2.bitwise_or(
        final_mask[y1:y2, x1:x2], mask_region)

    return final_mask
