from enum import Enum
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from einops import rearrange
import os
from typing import List, Tuple
import gdown
import gradio as gr
from PIL import Image
import sys
import subprocess
sys.path.append('ProPainter')


class InpaintMode(Enum):
  STTN = "STTN"
  LAMA = "LAMA"


class VideoInpainter:
  def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
    self.device = device

  def create_mask(self, frame: np.ndarray, height_percent: float = 0.2) -> np.ndarray:
    """Create a mask for the bottom portion of the frame"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    height = frame.shape[0]
    mask[int(height*(1-height_percent)):, :] = 255
    return mask

  def process_video(self, video_path: str, output_path: str, height_percent: float = 0.2, progress=gr.Progress()):
    """Process the video using OpenCV's inpainting"""
    # Read video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps,
                          (int(cap.get(3)), int(cap.get(4))))

    frame_count = 0
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      # Create mask for this frame
      mask = self.create_mask(frame, height_percent)

      # Apply inpainting
      inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

      # Write processed frame
      out.write(inpainted)

      frame_count += 1
      progress(frame_count/total_frames, desc="Processing frames...")

    cap.release()
    out.release()

    return output_path
