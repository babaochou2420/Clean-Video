import enum


class ModelEnum(enum.Enum):
  OPENCV = "[CPU] OpenCV"
  LAMA = "[GPU] LAMA"
  STTN = "[GPU] STTN"

  @classmethod
  def values(cls):
    """
    Returns a list of all the values in the ModelEnum.
    """
    return [item.value for item in cls]
