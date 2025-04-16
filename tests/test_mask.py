import cv2
import numpy as np
import os
import torch
from tqdm import tqdm
from daos.TextDetector import TextDetector
from utils.logger import setup_logger
import enum
# Assume your VideoInpainter class and Config are defined as in your original code

# Setup logger
logger = setup_logger('video_inpainter_test')


class MaskMode(enum.Enum):
  TEXT = "Text (Subtitles)"
  WATERMARK = "Watermark"


# class VideoInpainterTest:
#   def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
#     self.device = device
#     self.logger = logger
#     self.logger.info("VideoInpainterTest initialized")
#     self.text_detector = TextDetector()

#   def create_mask(self, frame: np.ndarray, mask_mode: MaskMode = MaskMode.TEXT) -> np.ndarray:
#     """Create a mask depending on the selected mode"""
#     self.logger.debug(f"Creating mask with mode: {mask_mode}")

#     if mask_mode == MaskMode.TEXT:
#       return self.text_detector.create_subtitle_mask(frame)
#     elif mask_mode == MaskMode.WATERMARK:
#       self.logger.warning(
#           "Watermark mask mode not implemented yet, returning empty mask")
#       return np.zeros(frame.shape[:2], dtype=np.uint8)

#   def test_mask_generation(self, video_path: str, output_folder: str, num_frames: int = 5):
#     """
#     Tests the mask generation process by saving a few frames and their corresponding masks.

#     Args:
#         video_path (str): Path to the input video file.
#         output_folder (str): Path to the folder where test frames and masks will be saved.
#         num_frames (int): Number of frames to process for testing.
#     """
#     self.logger.info(f"Starting mask generation test for video: {video_path}")
#     os.makedirs(output_folder, exist_ok=True)

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#       self.logger.error("Could not open video file")
#       return

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
#     processed_count = 0

#     with tqdm(total=num_frames, desc="Testing Mask Generation") as pbar:
#       current_frame_index = 0
#       while cap.isOpened() and processed_count < num_frames:
#         ret, frame = cap.read()
#         if not ret:
#           break

#         if current_frame_index in frame_indices:
#           mask = self.create_mask(frame, MaskMode.TEXT)

#           frame_filename = os.path.join(
#               output_folder, f"frame_{processed_count}.png")
#           mask_filename = os.path.join(
#               output_folder, f"mask_{processed_count}.png")

#           cv2.imwrite(frame_filename, frame)
#           # Scale mask to 0-255 for visualization
#           cv2.imwrite(mask_filename, mask)

#           self.logger.info(
#               f"Saved frame and mask: {frame_filename}, {mask_filename}")
#           processed_count += 1
#           pbar.update(1)

#         current_frame_index += 1

#     cap.release()
#     self.logger.info(
#         f"Finished mask generation test. Results saved in: {output_folder}")


if __name__ == '__main__':
  # Example usage: Replace 'your_video.mp4' with the actual path to your video file
  video_path = 'test.mp4'
  output_test_folder = 'test_mask_output'

  # Create an instance of the test class
  # tester = VideoInpainterTest()
  # tester.test_mask_generation(video_path, output_test_folder, num_frames=5)
