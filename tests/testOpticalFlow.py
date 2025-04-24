import cv2
import numpy as np
import matplotlib.pyplot as plt
from daos.image.OpticalFlow import FlowGuidedInpainter

# Load input images
original = cv2.imread("tests/output/frames/frame_0360.png")
inpainted = cv2.imread("tests/output/inpainted/inpaint_0360.png")
mask = cv2.imread("tests/output/masks/mask_0360.png", cv2.IMREAD_GRAYSCALE)

# Initialize the enhancer
enhancer = FlowGuidedInpainter()
final = enhancer.enhance(original, inpainted, mask)

# Save the result
cv2.imwrite("tests/output/opticalflow/enhanced_0360.png", final)

# --- Optical flow visualization ---


def draw_optical_flow(flow):
  h, w = flow.shape[:2]
  hsv = np.zeros((h, w, 3), dtype=np.uint8)

  mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
  hsv[..., 0] = ang * 180 / np.pi / 2  # Hue: direction
  hsv[..., 1] = 255                    # Saturation: constant
  hsv[..., 2] = cv2.normalize(
      mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude

  return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


flow = enhancer.compute_flow(original, inpainted)
flow_img = draw_optical_flow(flow)

# --- Matplotlib Visualization ---
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")

axes[1].imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
axes[1].set_title("Inpainted")

axes[2].imshow(flow_img)
axes[2].set_title("Optical Flow")

axes[3].imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
axes[3].set_title("Enhanced Output")

for ax in axes:
  ax.axis("off")

plt.tight_layout()
plt.savefig("tests/output/opticalflow/visual_comparison_0360.png", dpi=150)
plt.show()
