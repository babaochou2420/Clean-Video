# SDInpaint.py

import torch
import numpy as np
from PIL import Image
from diffusers import AutoPipelineForInpainting

from utils.logger import setup_logger


class SDXL:
  def __init__(self, seed=42, device="cuda" if torch.cuda.is_available() else "cpu"):
    self.device = device
    self.logger = setup_logger(__name__)
    self.pipeline = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        variant="fp16" if self.device == "cuda" else None
    )

    if self.device == "cuda":
      self.pipeline.enable_model_cpu_offload()
      try:
        self.pipeline.enable_xformers_memory_efficient_attention()
      except Exception as e:
        print("[Warning] xFormers not enabled:", e)

    self.generator = torch.Generator(self.device).manual_seed(seed)

  def process(self, image: np.ndarray, mask: np.ndarray, prompt: str = "") -> np.ndarray:
    init_image = Image.fromarray(image).convert("RGB")
    mask_image = Image.fromarray(mask).convert("RGB")

    try:
      result_image = self.pipeline(
          prompt=prompt,
          image=init_image,
          mask_image=mask_image,
          generator=self.generator
      ).images[0]
    except Exception as e:
      self.logger.error(f"Error: {e}")

    return np.array(result_image), mask
