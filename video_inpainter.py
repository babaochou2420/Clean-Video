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
from daos.TextDetector import TextDetector


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

    self.config = Config.get_config()

    self.maskHelper = MaskHelper()

  @log_function(logger)
  def create_mask(self, frame: np.ndarray, mask_mode: MaskMode = MaskMode.TEXT) -> np.ndarray:
    """Create a mask depending on the selected mode"""
    self.logger.debug(f"Creating mask with mode: {mask_mode}")

    if mask_mode == MaskMode.TEXT:
      return self.maskHelper.maskSubtitle(frame)
    elif mask_mode == MaskMode.WATERMARK:
      # Placeholder for future watermark detection

      self.logger.warning(
          "Watermark mask mode not implemented yet, returning empty mask")

      return np.zeros(frame.shape[:2], dtype=np.uint8)

  @log_function(logger)
  def processFrame(self, frame: np.ndarray, model: str, enableFTransform: bool = False, inpaintRadius: int = 3) -> Optional[np.ndarray]:
    """Process a single frame with the specified model and parameters"""
    try:
      match model:
        case ModelEnum.LAMA.value:
          mask = self.maskHelper.maskSubtitleBBoxes(frame)

          return self.lama.__call__(frame, mask), mask
        case ModelEnum.OPENCV.value:
          mask = self.maskHelper.maskSubtitle(frame)

          if enableFTransform:
            result = self.fast_ft_inpaint(
                frame, mask, inpaintRadius, cv2.ft.LINEAR, cv2.ft.ITERATIVE)
            # return cv2.ft.inpaint(frame, mask, inpaintRadius, function=cv2.ft.LINEAR, algorithm=cv2.ft.ITERATIVE)
          else:
            result = cv2.inpaint(frame, mask, inpaintRadius, cv2.INPAINT_TELEA)

          # Stage 2
          cv2.inpaint(frame, mask, inpaintRadius, cv2.INPAINT_NS)

          return result, mask

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

    self.loadModel(model)

    with tqdm(total=total_frames) as pbar:
      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          self.logger.info("End of video reached")
          break

        inpainted, mask = self.processFrame(frame, model, enableFTransform)
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
      from daos.inpainter.LAMA import LAMA

      self.lama = LAMA(device="cuda")

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
    pad = 0
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(frame.shape[1], x_max + pad)
    y_max = min(frame.shape[0], y_max + pad)

    # Calculate region dimensions
    region_width = x_max - x_min
    region_height = y_max - y_min

    # Determine optimal tile size (aim for tiles around 256x256)
    tile_size = 256
    overlap = 32  # Overlap between tiles for blending

    # Calculate number of tiles needed
    num_tiles_x = (region_width + tile_size - 1) // tile_size
    num_tiles_y = (region_height + tile_size - 1) // tile_size

    # Create result array
    result = frame.copy()

    # Convert frame to grayscale for edge detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create CUDA stream for parallel processing
    stream = cv2.cuda_Stream()

    # Upload frame to GPU
    gpu_frame = cv2.cuda_GpuMat()
    gpu_frame.upload(gray_frame, stream)

    # Multi-scale edge detection
    edges_multi_scale = []
    scales = [1.0, 0.75, 0.5]  # Different scales for edge detection

    for scale in scales:
        # Resize frame for current scale
      scaled_width = int(gray_frame.shape[1] * scale)
      scaled_height = int(gray_frame.shape[0] * scale)
      gpu_scaled = cv2.cuda.resize(gpu_frame, (scaled_width, scaled_height))

      # Create CUDA Canny detector with adaptive thresholds
      canny = cv2.cuda.createCannyEdgeDetector(100, 200)
      gpu_edges = canny.detect(gpu_scaled, stream)

      # Download and resize edges back to original size
      edges = gpu_edges.download(stream)
      edges = cv2.resize(edges, (gray_frame.shape[1], gray_frame.shape[0]))
      edges_multi_scale.append(edges)

    # Combine multi-scale edges
    combined_edges = np.zeros_like(gray_frame)
    for edges in edges_multi_scale:
      combined_edges = cv2.bitwise_or(combined_edges, edges)

    # Dilate edges with adaptive kernel size
    kernel_sizes = [3, 5, 7]
    dilated_edges = np.zeros_like(combined_edges)
    for ksize in kernel_sizes:
      kernel = np.ones((ksize, ksize), np.uint8)
      dilated = cv2.dilate(combined_edges, kernel, iterations=1)
      dilated_edges = cv2.bitwise_or(dilated_edges, dilated)

    # Combine edges with mask using guided filtering
    edge_near_mask = cv2.bitwise_and(dilated_edges, mask)
    guided_mask = cv2.bitwise_or(mask, edge_near_mask)

    # Apply guided filter to smooth the mask while preserving edges
    guided_mask = cv2.ximgproc.guidedFilter(
        guide=gray_frame,
        src=guided_mask.astype(np.float32),
        radius=5,
        eps=0.01
    ).astype(np.uint8)

    # Process each tile
    for i in range(num_tiles_y):
      for j in range(num_tiles_x):
          # Calculate tile boundaries
        tile_x_min = x_min + j * (tile_size - overlap)
        tile_y_min = y_min + i * (tile_size - overlap)
        tile_x_max = min(x_max, tile_x_min + tile_size)
        tile_y_max = min(y_max, tile_y_min + tile_size)

        # Adjust for last tiles
        if tile_x_max == x_max:
          tile_x_min = max(x_min, tile_x_max - tile_size)
        if tile_y_max == y_max:
          tile_y_min = max(y_min, tile_y_max - tile_size)

        # Extract tile
        tile_frame = frame[tile_y_min:tile_y_max, tile_x_min:tile_x_max]
        tile_mask = 255 - \
            guided_mask[tile_y_min:tile_y_max, tile_x_min:tile_x_max]

        # Calculate texture complexity
        tile_gray = gray_frame[tile_y_min:tile_y_max, tile_x_min:tile_x_max]
        tile_edges = dilated_edges[tile_y_min:tile_y_max,
                                   tile_x_min:tile_x_max]

        # Calculate edge density and texture variance
        edge_density = np.sum(tile_edges > 0) / (tile_edges.size + 1e-6)
        texture_variance = np.var(tile_gray)

        # Adaptive radius based on both edge density and texture complexity
        base_radius = max(
            3, min(radius, int(min(tile_frame.shape[:2]) * 0.05)))
        complexity_factor = 1 + edge_density + (texture_variance / 1000)
        tile_radius = int(base_radius / complexity_factor)

        # Inpaint tile
        inpainted_tile = cv2.ft.inpaint(
            tile_frame, tile_mask, tile_radius,
            function=function, algorithm=algorithm
        )

        # Create blending mask with texture-aware weighting
        blend_mask = np.ones_like(tile_frame, dtype=np.float32)
        if overlap > 0:
          # Create smooth blending at edges
          blend_width = min(overlap, tile_frame.shape[1] // 4)
          blend_height = min(overlap, tile_frame.shape[0] // 4)

          # Create texture-aware weights
          texture_weights = np.zeros_like(tile_frame, dtype=np.float32)
          # Stronger preservation at edges
          texture_weights[tile_edges > 0] = 0.7

          # Add texture variance influence
          texture_variance_map = np.abs(tile_gray - np.mean(tile_gray)) / 255.0
          texture_weights += texture_variance_map[..., None] * 0.3

          # Horizontal blending
          if j > 0:  # Left edge
            blend_mask[:,
                       :blend_width] *= np.linspace(0, 1, blend_width)[None, :, None]
          if j < num_tiles_x - 1:  # Right edge
            blend_mask[:, -
                       blend_width:] *= np.linspace(1, 0, blend_width)[None, :, None]

          # Vertical blending
          if i > 0:  # Top edge
            blend_mask[:blend_height,
                       :] *= np.linspace(0, 1, blend_height)[:, None, None]
          if i < num_tiles_y - 1:  # Bottom edge
            blend_mask[-blend_height:,
                       :] *= np.linspace(1, 0, blend_height)[:, None, None]

          # Apply texture-aware weights
          blend_mask = blend_mask * (1 - texture_weights) + texture_weights

        # Blend the inpainted tile into the result
        result[tile_y_min:tile_y_max, tile_x_min:tile_x_max] = (
            result[tile_y_min:tile_y_max, tile_x_min:tile_x_max] * (1 - blend_mask) +
            inpainted_tile * blend_mask
        ).astype(np.uint8)

    return result

  #
  #
  # [Args]
  # i:str - Video path
  # i:str - Model name
  # i:bool - Enable FTransform
  # i:int - Inpaint radius
  # o:np.ndarray - Inpainted frame
  # o:np.ndarray - Mask
  #
  def genPreview(self, video_path: str, model: str, enableFTransform: bool = False, inpaintRadius: int = 3) -> Tuple[np.ndarray, np.ndarray]:

    self.loadModel(model)

    # Open video and get random frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
      raise RuntimeError("Could not open video file")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_idx = np.random.randint(0, total_frames)

    self.logger.debug(
        f"[RUN] genPreview | Working on frame {random_frame_idx}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
      raise RuntimeError("Could not read frame from video")

    # Generate mask and process the frame
    preview, mask = self.processFrame(
        frame, model, enableFTransform, inpaintRadius)

    maskOverlay = MaskHelper.maskOverlay(frame, mask)

    cv2.imwrite("preview_maskOverlay.png", maskOverlay)
    cv2.imwrite("preview_inpainted.png", preview)

    self.logger.debug(f"[END] genPreview")

    return maskOverlay, preview
