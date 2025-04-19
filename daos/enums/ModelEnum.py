import enum
import cv2


class ModelEnum(enum.Enum):
  OPENCV = "[CPU] OpenCV"
  LAMA = "[GPU] LAMA"
  STTN = "[GPU] STTN"
  SDXL = "[GPU] SDXL"

  # EASYOCR = "[GPU] EasyOCR"
  # DB50 = "[CPU] DB50"
  # DB18 = "[CPU] DB18"

  @classmethod
  def values(cls):
    return [item.value for item in cls]


class OpenCVFTEnum(enum.Enum):
  ONE_STEP = "ONE_STEP"
  MULTI_STEP = "MULTI_STEP"
  ITERATIVE = "ITERATIVE"

  @classmethod
  def values(cls):
    return [item.value for item in cls]


class OCRModelEnum(enum.Enum):
  easyocr = "Easy-OCR"
  paddleocr = "PaddleOCR"

  @classmethod
  def values(cls):
    return [item.value for item in cls]
