import cv2
import os
from daos.MaskHelper import MaskHelper
from utils.logger import setup_logger
from video_inpainter import VideoInpainter

# Setup logger
logger = setup_logger('gen_test')


def process_video():
  # Create output directories
  output_dirs = {
      'frames': 'tests/output/frames',
      'masks': 'tests/output/masks',
      'inpainted': 'tests/output/inpainted'
  }
  for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

  # Initialize mask helper
  mask_helper = MaskHelper()

  # Open video
  video_path = 'test.mp4'
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    logger.error(f"Could not open video: {video_path}")
    return

  # Get video properties
  fps = cap.get(cv2.CAP_PROP_FPS)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  logger.info(f"Processing video: {video_path}")
  logger.info(f"FPS: {fps}, Total frames: {total_frames}")

  frame_count = 0
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    if frame_count % 60 == 0:
      logger.info(f"Processing frame {frame_count}/{total_frames}")

      # Save original frame
      frame_path = os.path.join(
          output_dirs['frames'], f'frame_{frame_count:04d}.png')
      cv2.imwrite(frame_path, frame)

      # Generate and save mask
      mask = mask_helper.maskSubtitle(frame)
      mask_path = os.path.join(
          output_dirs['masks'], f'mask_{frame_count:04d}.png')
      cv2.imwrite(mask_path, mask)

      # Generate and save visualization
      videoInpainter = VideoInpainter()
      vis = videoInpainter.processFrame(frame, mask, "[CPU] OpenCV")
      vis_path = os.path.join(
          output_dirs['inpainted'], f'inpaint_{frame_count:04d}.png')
      cv2.imwrite(vis_path, vis)

    frame_count += 1

  cap.release()
  logger.info("Processing complete!")


if __name__ == '__main__':
  process_video()
