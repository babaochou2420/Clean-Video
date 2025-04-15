from enum import Enum
from multiprocessing import Pool
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
from daos.MaskHelper import MaskHelper
from utils.logger import setup_logger, log_function
from tqdm import tqdm

from daos.enums.ModelEnum import ModelEnum

from utils.config import Config
from daos.text_detector import TextDetector

config = Config.get_config()

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
  def create_mask(self, frame: np.ndarray, mask_mode: MaskMode = MaskMode.TEXT) -> np.ndarray:
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
  def processFrame(self, frame: np.ndarray, mask: np.ndarray, model: str, enableFTransform: bool = False, inpaintRadius: int = 3) -> Optional[np.ndarray]:
    """Process a single frame with the specified model and parameters"""
    try:
      match model:
        case ModelEnum.LAMA.value:
          self.loadModel(model)
          return self.lama.process(frame, mask)
        case ModelEnum.OPENCV.value:
          if enableFTransform:
            return cv2.ft.inpaint(frame, mask, inpaintRadius, function=cv2.ft.LINEAR, algorithm=cv2.ft.ONE_STEP)
          else:
            return cv2.inpaint(frame, mask, inpaintRadius, cv2.INPAINT_TELEA)
        case _:
          logger.error(f"Unknown model selected: {model}")
          return None
    except Exception as e:
      self.logger.error(f"Error processing frame: {e}")
      return None

  @log_function(logger)
  def process_video(self, video_path: str, output_path: str, model: str, enableFTransform: bool = False, mask_mode: MaskMode = MaskMode.TEXT) -> Optional[str]:
    """Process the video and apply inpainting on detected text regions"""
    self.logger.info(
        f"Start video processing with [{model}] on mask mode: {mask_mode}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      self.logger.error("Could not open video file")
      return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    self.logger.info(f"{width}x{height} @ {fps}fps | {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
      self.logger.error("Could not create output video file")
      return None

    with tqdm(total=total_frames) as pbar:
      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          self.logger.info("End of video reached")
          break

        mask = self.create_mask(frame)
        if mask is None:
          self.logger.error("Failed to create mask")
          continue

        inpainted = self.processFrame(frame, mask, model, enableFTransform)
        if inpainted is None:
          self.logger.error("Failed to process frame")
          continue

        out.write(inpainted)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    self.logger.info(f"Processing complete. Output saved to: {output_path}")

    return output_path

  def loadModel(self, model: str):
    if model == ModelEnum.LAMA.value:
      from daos.models.LAMA import LAMA

      self.lama = LAMA(config["models"]["LAMA"]["ckpt"],
                       config["models"]["LAMA"]["config"], device="CUDA")
    # elif model == "STTN":
    #   from daos.models.STTN import STTN
    #   return STTN()

  #
  # [LOGIC]
  # 1. Inpaint the cropped region instead of whole resolution which cause MULTI_STEP and ITERATIVE to take forever
  # 2. Paste the inpainted region back to the frame
  # 3. Return the result
  #
  def fast_ft_inpaint(self,
                      frame, mask, radius=3, function=cv2.ft.LINEAR, algorithm=cv2.ft.ONE_STEP):
    # Step 1: Get bounding box of non-zero region
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        # Nothing to inpaint
      return frame.copy()

    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()

    # Add padding to avoid hard seams
    pad = 16
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(frame.shape[1], x_max + pad)
    y_max = min(frame.shape[0], y_max + pad)

    # Step 2: Crop
    cropped_frame = frame[y_min:y_max, x_min:x_max]
    # Invert the mask so white (255) becomes the area to inpaint
    cropped_mask = 255 - mask[y_min:y_max, x_min:x_max]

    # Step 3: Inpaint small region
    inpainted_crop = cv2.ft.inpaint(
        cropped_frame, cropped_mask, radius, function=function, algorithm=algorithm)

    # TEST
    # Sample 5 frames for testing
    test_output_dir = "test_ft_inpaint_output"
    os.makedirs(test_output_dir, exist_ok=True)

    frame_num = self.frame_count if hasattr(self, 'frame_count') else 0
    frame_name = f"frame_{frame_num}"

    cv2.imwrite(os.path.join(test_output_dir,
                f"{frame_name}_inpainted_crop.png"), inpainted_crop)
    cv2.imwrite(os.path.join(test_output_dir,
                f"{frame_name}_cropped_frame.png"), cropped_frame)
    cv2.imwrite(os.path.join(test_output_dir,
                f"{frame_name}_cropped_mask.png"), cropped_mask)

    # Step 4: Paste back
    result = frame.copy()
    result[y_min:y_max, x_min:x_max] = inpainted_crop

    return result

  def genPreview(self, video_path: str, model: str, enableFTransform: bool = False, inpaintRadius: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a preview of the inpainting process by processing a random frame from the video.
    Returns the original frame, generated mask, and processed result.

    Args:
        video_path (str): Path to the input video file
        output_folder (str): Path to save the preview results

    Returns:
        tuple: (original_frame, mask, processed_frame) as numpy arrays
    """

    # Open video and get random frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      raise RuntimeError("Could not open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_idx = np.random.randint(0, total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
      raise RuntimeError("Could not read frame from video")

    # Generate mask and process the frame
    mask = self.create_mask(frame)
    preview = self.processFrame(
        frame, mask, model, enableFTransform, inpaintRadius)

    maskOverlay = MaskHelper.maskOverlay(frame, mask)

    return maskOverlay, preview
