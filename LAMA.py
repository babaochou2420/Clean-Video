from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.modules import make_generator
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


class LAMA:
  def __init__(self, model_path):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = make_generator().to(self.device)
    self.model.eval()

    # Load the checkpoint
    checkpoint = load_checkpoint(model_path, self.device)
    self.model.load_state_dict(checkpoint['state_dict'])

  def __call__(self, image, mask):
    """
    Process an image with a mask using the LAMA model.

    Args:
        image (torch.Tensor): Input image tensor of shape [B, C, H, W]
        mask (torch.Tensor): Binary mask tensor of shape [B, 1, H, W]

    Returns:
        torch.Tensor: Inpainted image tensor
    """
    with torch.no_grad():
      # Ensure inputs are on the correct device
      image = image.to(self.device)
      mask = mask.to(self.device)

      # Forward pass
      output = self.model(image, mask)

      # Combine original image and inpainted result
      result = image * (1 - mask) + output * mask

      return result
