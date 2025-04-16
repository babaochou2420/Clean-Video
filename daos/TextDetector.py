import os
import cv2
import numpy as np
import easyocr
from typing import List, Tuple, Literal
from utils.config import Config
from utils.logger import setup_logger, log_function

logger = setup_logger('text_detector')

OCRBackend = Literal['easyocr', 'east', 'db']


class TextDetector:
  def __init__(self, backend: OCRBackend = 'easyocr', lang_list=None):
    self.config = Config().get_config()
    logger.info(f"Initializing TextDetector with backend: {backend}")
    self.backend = backend
    self.reader = None

    if backend == 'easyocr':
      self.reader = easyocr.Reader(lang_list or ['en'], gpu=True)
    elif backend in ['east', 'db']:
      self.reader = self._load_opencv_model(backend)
    else:
      raise ValueError(f"Unsupported backend: {backend}")

  def _load_opencv_model(self, model: str):
    if model == 'east':
      model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'models', 'frozen_east_text_detection.pb')
      if not os.path.exists(model_path):
        logger.warning("EAST model not found, downloading...")
        import urllib.request
        import tarfile

        # Download and extract model
        tar_path = model_path + '.tar.gz'
        urllib.request.urlretrieve(
            "https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz?dl=1",
            tar_path)

        # Extract tar.gz file
        with tarfile.open(tar_path) as tar:
          tar.extractall(os.path.dirname(model_path))

        # Remove tar.gz file
        os.remove(tar_path)

      return cv2.dnn.TextDetectionModel_EAST(model_path)
    elif model == 'db':
      modelPath = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               self.config["models"]['BASE']['TextDetection'], self.config["models"]['PATH']['TextDetection']['DB50'])
      if not os.path.exists(modelPath):
        logger.warning("DB model not found, downloading...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://drive.google.com/uc?export=dowload&id=19YWhArrNccaoSza0CfkXlA8im4-lAGsR",
            modelPath)

      return cv2.dnn.TextDetectionModel_DB(modelPath)
    return None

  @log_function(logger)
  def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect text and return list of bounding boxes."""
    if self.backend == 'easyocr':
      results = self.reader.readtext(image)
      return [result[0] for result in results]

    if self.backend in ['east', 'db']:
      return self.__detect_OpenCV(image)

  def __detect_OpenCV(self, image: np.ndarray):
    if self.backend == 'db':
      # Ensure input size is compatible with DB model (multiples of 32)
      h, w = image.shape[:2]
      new_w = (w // 32) * 32
      new_h = (h // 32) * 32

      # Set input parameters
      self.reader.setBinaryThreshold(0.3)
      self.reader.setPolygonThreshold(0.5)
      self.reader.setInputParams(
          scale=1.0,
          size=(new_w, new_h),
          mean=(123.675, 116.28, 103.53),  # Standard ImageNet mean
          swapRB=True
      )

    boxes, confidences = self.reader.detect(image)
    return boxes

  @log_function(logger)
  def visualize_detection(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay mask on image as red area for visualization."""
    red_mask = np.zeros_like(image)
    red_mask[:, :, 2] = mask
    vis = cv2.addWeighted(image, 0.7, red_mask, 0.3, 0)
    return vis
