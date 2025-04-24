import cv2
import numpy as np


class FlowGuidedInpainter:
  def __init__(self, flow_method='farneback', blur_ksize=5, alpha=0.7):
    """
    :param flow_method: 'farneback' or 'tvl1'
    :param blur_ksize: Kernel size for directional blur
    :param alpha: Blending weight between inpainted and flow-guided value
    """
    self.flow_method = flow_method
    self.blur_ksize = blur_ksize
    self.alpha = alpha

  def compute_flow(self, img1, img2):
    """Compute optical flow from img1 to img2."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if self.flow_method == 'farneback':
      flow = cv2.calcOpticalFlowFarneback(
          gray1, gray2, None,
          pyr_scale=0.5, levels=3, winsize=15,
          iterations=3, poly_n=5, poly_sigma=1.2, flags=0
      )
    elif self.flow_method == 'tvl1':
      tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
      flow = tvl1.calc(gray1, gray2, None)
    else:
      raise ValueError("Unsupported flow method")
    return flow

  def directional_blur(self, image, flow, mask):
    """
    Apply flow-guided blur in the direction of motion.
    Only applied to the masked area.
    """
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    mask = mask.astype(bool)

    for y in range(h):
      for x in range(w):
        if not mask[y, x]:
          continue

        dx, dy = flow[y, x]
        nx = int(np.clip(x + dx, 0, w - 1))
        ny = int(np.clip(y + dy, 0, h - 1))

        color_src = image[ny, nx].astype(np.float32)
        color_dst = result[y, x]
        result[y, x] = self.alpha * color_src + (1 - self.alpha) * color_dst

    return np.clip(result, 0, 255).astype(np.uint8)

  def enhance(self, original_frame, inpainted_frame, mask):
    """
    Main entrypoint: enhance the inpainted result using optical flow.
    :param original_frame: Original frame (before inpainting)
    :param inpainted_frame: Result from e.g., cv2.inpaint or LaMa
    :param mask: Binary mask of the inpainted region (uint8, 0/255)
    :return: Enhanced image
    """
    assert original_frame.shape == inpainted_frame.shape, "Size mismatch"
    assert mask.ndim == 2, "Mask should be single-channel"

    flow = self.compute_flow(original_frame, inpainted_frame)
    result = self.directional_blur(inpainted_frame, flow, mask)
    return result
