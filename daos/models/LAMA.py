import torch
import cv2
import numpy as np
import os
from PIL import Image
import yaml  # Import the PyYAML library


class LAMA:
  """
  Python class to encapsulate the LAMA inpainting model.

  Attributes:
      model_path (str): Path to the LAMA model checkpoint.
      device (torch.device): The device to run the model on (CPU or GPU).
      model (torch.nn.Module): The loaded LAMA model.
      config (dict): Configuration parameters for the model, loaded from a YAML file.
  """

  def __init__(self, model_path, config_path, device='cuda'):
    """
    Initializes the LAMA class.  Loads the model and configuration, and sets up the device.

    Args:
        model_path (str): Path to the LAMA model checkpoint (e.g., 'path/to/lama.pth').
        config_path (str): Path to the LAMA configuration YAML file (e.g., 'path/to/config.yaml').
        device (str, optional): 'cuda' or 'cpu'. Defaults to 'cuda'.
    """
    self.model_path = model_path
    self.device = torch.device(device) if torch.cuda.is_available(
    ) and device == 'cuda' else torch.device('cpu')
    self.model = None
    self.config = self.load_config(config_path)  # Load config from YAML
    self.load_model()

  def load_config(self, config_path):
    """
    Loads the configuration from the specified YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        YAMLError: If there is an error parsing the YAML file.
    """
    if not os.path.exists(config_path):
      raise FileNotFoundError(f"Configuration file not found at {config_path}")
    try:
      with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
      return config
    except yaml.YAMLError as e:
      raise yaml.YAMLError(f"Error parsing YAML config: {e}")

  def load_model(self):
    """
    Loads the LAMA model from the specified checkpoint, using the configuration.

    Raises:
        FileNotFoundError: If the model path does not exist.
        Exception: For other errors during model loading.
    """
    if not os.path.exists(self.model_path):
      raise FileNotFoundError(
          f"Model checkpoint not found at {self.model_path}")

    try:
      print(f"Loading LAMA model from {self.model_path}...")
      checkpoint = torch.load(self.model_path, map_location=self.device)

      #  This is where you would use the configuration to create the model.
      #  The example below is a placeholder.  You **MUST** replace it with
      #  the actual model creation code from the LAMA repository.  The LAMA
      #  code typically defines the model class and then loads the weights
      #  from the checkpoint.  The configuration dictionary (`self.config`)
      #  will contain the necessary parameters for model creation.
      #
      # Example (VERY LIKELY WRONG - REPLACE):
      # from lama_model import LaMaModel  #  <--  Import the model class
      # self.model = LaMaModel(self.config) # <-- Use config here
      # self.model.load_state_dict(checkpoint['model']) # Or 'state_dict', etc.
      # self.model.to(self.device)
      # self.model.eval()
      #
      #  The correct code will look something like the above, but you need
      #  to adapt it to *exactly* match the LAMA code.  Look at their
      #  model definition and loading logic.

      # Placeholder for the correct LAMA model loading.  This is CRITICAL to replace.
      if 'model' in checkpoint:
        self.model = checkpoint['model']
      elif 'state_dict' in checkpoint:
        self.model = checkpoint['state_dict']
      else:
        self.model = checkpoint  # hope for the best.
      self.model.to(self.device)
      self.model.eval()

      print("LAMA model loaded successfully.  **IMPORTANT:** You MUST replace the placeholder loading with the correct LAMA loading code, using the parameters from the loaded configuration.")

    except Exception as e:
      raise Exception(f"Error loading model: {e}")

  def process(self, image, mask):
    """
    Performs inpainting on the given image using the provided mask and the model configuration.

    Args:
        image (numpy.ndarray): The input image as a NumPy array (H, W, 3) in RGB format,
                               or a PIL Image.
        mask (numpy.ndarray): The mask as a NumPy array (H, W, 1) or (H, W) where 1 indicates
                              the masked region, or a PIL Image.

    Returns:
        numpy.ndarray: The inpainted image as a NumPy array (H, W, 3) in RGB format.

    Raises:
        ValueError: If the image and mask have incompatible shapes or types.
        RuntimeError: If the model fails to process the input.
    """
    if isinstance(image, Image.Image):
      image = np.array(image.convert('RGB'))
    if isinstance(mask, Image.Image):
      mask = np.array(mask.convert('L'))

    if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
      raise ValueError("Image and mask must be NumPy arrays or PIL Images.")

    if image.ndim != 3 or image.shape[2] != 3:
      raise ValueError(
          "Image must be a NumPy array with shape (H, W, 3) in RGB format.")
    if mask.ndim == 2:
      mask = mask[:, :, np.newaxis]
    if mask.ndim != 3 or mask.shape[2] != 1:
      raise ValueError(
          "Mask must be a NumPy array with shape (H, W, 1) or (H,W).")

    if image.shape[:2] != mask.shape[:2]:
      raise ValueError("Image and mask must have the same height and width.")

    h, w = image.shape[:2]
    # Use image_size from config
    image = cv2.resize(
        image, (self.config['image_size'], self.config['image_size']))
    mask = cv2.resize(
        mask, (self.config['image_size'], self.config['image_size']), interpolation=cv2.INTER_NEAREST)

    img_tensor = torch.from_numpy(image.astype(
        np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
    mask_tensor = torch.from_numpy(mask.astype(
        np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
    mask_tensor = (mask_tensor > 0.5).float()

    try:
      with torch.no_grad():
        #  This is where you call the LAMA model.  You must adapt this
        #  to the specific input requirements of your LAMA model.
        #  The configuration is available in `self.config`.
        #
        # Example (VERY LIKELY WRONG - REPLACE):
        # output = self.model(img_tensor, mask_tensor, self.config['inpainting_params'])  # Use config
        #
        #  You need to use the keys and values from your configuration
        #  file here.  The LAMA code will expect specific arguments to
        #  its forward() method.

        # Placeholder for the actual LAMA model call.  Use self.config.
        output = self.model(img_tensor, mask_tensor)
        inpainted_image = output[0].cpu().permute(1, 2, 0).numpy()
        inpainted_image = (inpainted_image * 255).astype(np.uint8)

    except Exception as e:
      raise RuntimeError(f"Error during inpainting: {e}")

    inpainted_image = cv2.resize(inpainted_image, (w, h))
    return inpainted_image
