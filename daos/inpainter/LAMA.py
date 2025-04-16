import torch
import cv2
import numpy as np
import os
from PIL import Image
import yaml

from utils import logger
from utils.config import Config
from torchvision import transforms


import sys
from pathlib import Path

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent))


class LAMA:
  def __init__(self, model_path, config_path, device='cuda'):
    self.model_path = model_path
    self.config_path = config_path
    self.device = torch.device(device) if torch.cuda.is_available(
    ) and device == 'cuda' else torch.device('cpu')
    self.model = None
    self.transform = None
    self.load_model()

    self.logger = logger.setup_logger("LAMA")

  def load_model(self):
    if not os.path.exists(self.model_path):
      raise FileNotFoundError(
          f"Model checkpoint not found at {self.model_path}")
    if not os.path.exists(self.config_path):
      raise FileNotFoundError(f"Config file not found at {self.config_path}")

    try:
      print(f"Loading LAMA config from {self.config_path}...")
      with open(self.config_path, 'r') as f:
        config = yaml.safe_load(f)

      self.setup_transform(config)

      print("Building LAMA model...")
      from daos.inpainter.saicinpainting.training.trainers import load_checkpoint

      # Load model from checkpoint with full config and architecture
      train_config = config.get('model', config)
      self.model = load_checkpoint(
          train_config, self.model_path, strict=False, map_location=self.device)['model']
      self.model.freeze()  # turn off grad
      self.model.eval()
      self.model.to(self.device)

      print("LAMA model loaded and ready.")

    except Exception as e:
      raise RuntimeError(f"Error loading LAMA model: {e}")

  def setup_transform(self, config):
    # You can customize this as needed based on LaMa config
    self.transform = transforms.Compose([
        transforms.ToTensor(),
    ])

  def prepare_input(self, image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Converts an OpenCV image and binary mask to input format expected by LaMa.
    """
    if image.dtype != np.uint8:
      raise ValueError("Image must be uint8")

    if mask.ndim == 3:
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = (mask > 0).astype(np.uint8) * 255

    # Convert to RGB
    if image.shape[2] == 4:
      image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_pil = Image.fromarray(image)
    mask_pil = Image.fromarray(mask)

    image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
    mask_tensor = self.transform(mask_pil).unsqueeze(0).to(self.device)

    batch = {
        'image': image_tensor,
        'mask': mask_tensor
    }

    return batch

  def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Runs inpainting on the given image and mask.
    """
    with torch.no_grad():
      batch = self.prepare_input(image, mask)
      result = self.model(batch)['inpainted']

      result = result[0].clamp(0, 1).cpu().numpy()
      result = (result.transpose(1, 2, 0) * 255).astype(np.uint8)
      return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
