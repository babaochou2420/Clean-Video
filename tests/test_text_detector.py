import unittest
import os
import cv2
import numpy as np
from daos.text_detector import TextDetector


class TestTextDetector(unittest.TestCase):
  def setUp(self):
    self.detector = TextDetector()
    self.test_frames = [
        "test_frames/frame_004.png",
        "test_frames/frame_015.png",
        "test_frames/frame_030.png"
    ]

  def test_mask_on_real_frames(self):
    for frame_path in self.test_frames:
      with self.subTest(frame=frame_path):
        self.assertTrue(os.path.exists(frame_path),
                        f"File not found: {frame_path}")

        image = cv2.imread(frame_path)
        self.assertIsNotNone(image, f"Failed to read image: {frame_path}")

        mask = self.detector.create_subtitle_mask(image)

        # Check size match
        self.assertEqual(mask.shape, image.shape[:2], "Mask shape mismatch")

        # Check binary values
        unique = np.unique(mask)
        self.assertTrue(set(unique).issubset(
            {0, 255}), f"Non-binary mask values in {frame_path}: {unique}")

        # Check if mask has content
        self.assertGreater(np.sum(mask == 255), 0,
                           f"No text detected in {frame_path}")

  def test_visualization_output(self):
    os.makedirs("tests/output", exist_ok=True)

    for frame_path in self.test_frames:
      with self.subTest(frame=frame_path):
        image = cv2.imread(frame_path)
        mask = self.detector.create_subtitle_mask(image)
        overlay = self.detector.visualize_detection(image, mask)

        basename = os.path.basename(frame_path).replace(".png", "")
        cv2.imwrite(f"tests/output/{basename}_mask.png", mask)
        cv2.imwrite(f"tests/output/{basename}_overlay.png", overlay)


if __name__ == '__main__':
  unittest.main()
