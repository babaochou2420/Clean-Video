import os
import cv2
from daos.TextDetector import TextDetector

# === Config ===
backend = 'db'  # Change to 'east' or 'db' if needed
input_frames = [
    "test_frames/frame_004.png",
    "test_frames/frame_015.png",
    "test_frames/frame_030.png"
]
output_dir = "tests/sample/TextDetection"
os.makedirs(output_dir, exist_ok=True)

# === Init Detector ===
detector = TextDetector(backend=backend)

# === Run Detection ===
for path in input_frames:
  image = cv2.imread(path)
  if image is None:
    print(f"[ERROR] Could not read image: {path}")
    continue

  bboxes = detector.detect(image)

  # Draw boxes
  for box in bboxes:
    x, y, w, h = box
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

  filename = os.path.basename(path)
  output_path = os.path.join(output_dir, f"{filename}_{backend}.png")
  cv2.imwrite(output_path, image)
  print(f"[SAVED] {output_path}")
