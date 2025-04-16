import os
from typing import Union
import torch
import numpy as np
from PIL import Image
from daos.inpainter.lama_utils import prepare_img_and_mask
from utils import logger
from utils.config import Config


class LAMA:
  def __init__(self, device: torch.device = None):
    self.config = Config.get_config()
    self.device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    self.model_path = os.path.join(self.config['models']['BASE']['Inpainter'],
                                   self.config['models']['LAMA']['ckpt'])

    self.logger = logger.setup_logger(self.__class__.__name__)

    if not os.path.isfile(self.model_path):
      raise FileNotFoundError(f"Model not found at {self.model_path}")

    # checkpoint = torch.load(self.model_path, map_location=self.device)

    # self.logger.info(f"Loading LAMA model from {checkpoint}")

    # self.model = checkpoint['model']

    self.model = torch.jit.load(self.model_path, map_location=self.device)

    self.model.eval().to(self.device)

  def __call__(self, image: Union[Image.Image, np.ndarray], mask: Union[Image.Image, np.ndarray]) -> np.ndarray:
    orig_height, orig_width = np.array(image).shape[:2]

    image_tensor, mask_tensor = prepare_img_and_mask(image, mask, self.device)
    with torch.inference_mode():
      output = self.model(image_tensor, mask_tensor)[0]
      result = output.permute(1, 2, 0).cpu().numpy()
      result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
      return result[:orig_height, :orig_width]
