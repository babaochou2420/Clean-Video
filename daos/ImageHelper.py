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


def hex_to_rgb(hex_code: str):
  hex_code = hex_code.lstrip('#')
  return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def is_color_too_close(color_a, color_b, threshold=30):
  # Simple Euclidean distance in RGB
  return np.linalg.norm(np.array(color_a) - np.array(color_b)) < threshold


def filter_presets(preset_hex_list, dominant_colors_rgb, threshold=30):
  filtered = []
  for hex_color in preset_hex_list:
    rgb = hex_to_rgb(hex_color)
    if all(not is_color_too_close(rgb, dom, threshold) for dom in dominant_colors_rgb):
      filtered.append(hex_color)
  return filtered


# image = cv2.imread("sample.jpg")
# dominant_colors = get_dominant_colors(image, k=5)

# preset_colors = ["#FF0000", "#00FF00", "#FFFF00", "#0000FF"]
# available_colors = filter_presets(preset_colors, dominant_colors, threshold=40)

# print("Filtered preset colors:", available_colors)
