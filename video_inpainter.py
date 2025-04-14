from enum import Enum
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from einops import rearrange
import os
from typing import List, Tuple, Optional, Union
import gdown
import gradio as gr
from PIL import Image
import sys
import subprocess
from utils.logger import setup_logger, log_function
from tqdm import tqdm


from daos.text_detector import TextDetector


# Setup logger
logger = setup_logger('video_inpainter')


class MaskMode(Enum):
  TEXT = "Text (Subtitles)"
  WATERMARK = "Watermark"


class InpaintMode(Enum):
  STTN = "STTN"
  LAMA = "LAMA"


class VideoInpainter:
  def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
    self.device = device
    self.logger = logger
    self.logger.info("VideoInpainter initialized")
    self.text_detector = TextDetector()

  @log_function(logger)
  def create_mask(self, frame: np.ndarray, height_percent: float = 0.2, mask_mode: MaskMode = MaskMode.TEXT) -> np.ndarray:
    """Create a mask depending on the selected mode"""
    self.logger.debug(f"Creating mask with mode: {mask_mode}")

    if mask_mode == MaskMode.TEXT:
      return self.text_detector.create_subtitle_mask(frame)
    elif mask_mode == MaskMode.WATERMARK:
      # Placeholder for future watermark detection

      self.logger.warning(
          "Watermark mask mode not implemented yet, returning empty mask")

      return np.zeros(frame.shape[:2], dtype=np.uint8)
    # else:
    #   # Fallback to simple bottom height_percent
    #   height = frame.shape[0]
    #   mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    #   mask[int(height * (1 - height_percent)):, :] = 255
    #   self.logger.debug("Fallback bottom strip mask created")

  @log_function(logger)
  def process_video(self, video_path: str, output_path: str, height_percent: float = 0.2, mask_mode: MaskMode = MaskMode.TEXT, progress: Optional[gr.Progress] = None) -> Optional[str]:
    """Process the video and apply inpainting on detected text regions"""
    self.logger.info(f"Starting video processing: {video_path}")
    self.logger.debug(f"Output path: {output_path}")
    self.logger.debug(
        f"Height percent: {height_percent}, Mask mode: {mask_mode}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      self.logger.error("Could not open video file")
      return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    self.logger.info(
        f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
      self.logger.error("Could not create output video file")
      return None

    frame_count = 0
    with tqdm(total=total_frames) as pbar:
      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          self.logger.info("End of video reached")
          break

        mask = self.create_mask(frame)

        inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        out.write(inpainted)

        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    self.logger.info(f"Processing complete. Output saved to: {output_path}")
    self.logger.info(f"Total frames processed: {frame_count}")

    return output_path
