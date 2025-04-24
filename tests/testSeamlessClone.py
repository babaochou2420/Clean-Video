import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import logger

logger = logger.setup_logger("testSeamlessClone")

# Create output directory
os.makedirs("tests/output/seamless", exist_ok=True)

# Load images
original = cv2.imread("tests/output/frames/frame_0360.png")
inpainted = cv2.imread("tests/output/inpainted/inpaint_0360.png")
mask = cv2.imread("tests/output/masks/mask_0360.png", cv2.IMREAD_GRAYSCALE)

# Convert mask to binary
_, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

# Find center of the mask for seamless cloning
M = cv2.moments(mask_binary)
center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

# Try different seamless cloning methods
methods = [
    (cv2.NORMAL_CLONE, "Normal Clone"),
    (cv2.MIXED_CLONE, "Mixed Clone"),
    (cv2.MONOCHROME_TRANSFER, "Monochrome Transfer")
]

# Create figure for visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Show original and inpainted
axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title("Original Frame")
axes[0, 1].imshow(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title("Inpainted Frame")

# Try each method
results = []
for i, (method, name) in enumerate(methods):
  try:
    logger.info(center)
    # Apply seamless cloning
    result = cv2.seamlessClone(
        inpainted, original, mask_binary, center, method)

    # Save result
    output_path = f"tests/output/seamlessclone/{name.lower().replace(' ', '_')}_0360.png"
    cv2.imwrite(output_path, result)

    # Show result
    axes[1, i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(name)

    results.append((name, result))

  except Exception as e:
    print(f"Error with {name}: {str(e)}")

# Turn off axes
for ax in axes.flat:
  ax.axis('off')

# Save comparison plot
plt.tight_layout()
plt.savefig("tests/output/seamless/comparison_0360.png", dpi=150)
plt.show()

# Print results summary
print("\n=== Seamless Cloning Results ===")
for name, result in results:
  # Calculate difference metrics
  diff = cv2.absdiff(original, result)
  mse = np.mean(diff ** 2)
  psnr = 20 * np.log10(255.0 / np.sqrt(mse))

  print(f"\n{name}:")
  print(f"  MSE: {mse:.2f}")
  print(f"  PSNR: {psnr:.2f} dB")
  print(
      f"  Saved to: tests/output/seamless/{name.lower().replace(' ', '_')}_0360.png")
