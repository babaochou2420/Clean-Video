import cv2
from video_inpainter import VideoInpainter


VideoInpainter().genPreview(cv2.imread(
    "test_360.png"), "[GPU] LAMA")

# VideoInpainter().genPreview(cv2.imread(
#     "test.png"), "[GPU] LAMA")
# VideoInpainter().genPreview("test.mp4", "[GPU] STTN")
# VideoInpainter().genPreview("test.mp4", "[GPU] SDXL")
