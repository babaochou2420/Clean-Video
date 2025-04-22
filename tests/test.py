import os
import cv2
from daos.MaskHelper import MaskHelper
from video_inpainter import VideoInpainter

from daos.enums.ModelEnum import ModelEnum

modelName = ModelEnum.LAMA.value
# modelName = ModelEnum.OPENCV.value

saveDir = f"tests/sample/{modelName}"

os.makedirs(saveDir, exist_ok=True)

testers = [
    # "tests/frames/test1_500.png",
    # "tests/frames/test1_190.png",
    "tests/frames/test1_191.png",
    # "tests/frames/test1_192.png",
    #    "tests/frames/test2_210.png"
]

videoInpainter = VideoInpainter()

for path in testers:
  if not os.path.exists(path):
    print(f"File not found: {path}")
    continue

  maskOverlay, preview = videoInpainter.genPreview(
      cv2.imread(path), modelName)

  # Extract file name without extension
  base_filename = os.path.splitext(os.path.basename(path))[0]

  # Compose output paths
  out_inpainted = os.path.join(saveDir, f"{base_filename}_inpainted.png")
  out_mask_overlay = os.path.join(saveDir, f"{base_filename}_maskOverlay.png")

  # Save results
  cv2.imwrite(out_inpainted, preview)
  cv2.imwrite(out_mask_overlay, maskOverlay)

  print(f"Processed: {path}")
