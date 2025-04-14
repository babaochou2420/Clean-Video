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
  def process_video(self, video_path: str, output_path: str, model, enableFTransform: bool = False, mask_mode: MaskMode = MaskMode.TEXT) -> Optional[str]:
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

    self.logger.info(
        f"{width}x{height} @ {fps}fps | {total_frames} frames")

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

        mask = self.create_mask(frame, mask_mode)

        self.loadModel(model)

        match model:
          case ModelEnum.LAMA.value:
            inpainted = self.lama.process(frame, mask)

          case ModelEnum.OPENCV.value:

            def processFrame_OpenCV(enableFTransform, frame, mask, out):
              if enableFTransform:
                # MULTI_STEP and ITERATIVE will be taking forever for high resolution
                inpainted = cv2.ft.inpaint(
                    frame, mask, 3, function=cv2.ft.LINEAR, algorithm=cv2.ft.ONE_STEP)
              else:
                inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

              out.write(inpainted)

            with Pool(8) as pool:
              for idx in range(100):
                pool.apply_async(
                    processFrame_OpenCV,
                    (enableFTransform, frame, mask, out)
                )
              pool.close()
              pool.join()

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
