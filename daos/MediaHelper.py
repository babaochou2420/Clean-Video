import datetime
import json
import os
import random
import subprocess
import tempfile
from typing import List, Dict
import logging
from moviepy import VideoFileClip, AudioFileClip
import numpy as np


import cv2


class MediaHelper:
  def __init__(self):
    self.audio_cache = {}
    self.temp_dir = tempfile.gettempdir()
    self.logger = logging.getLogger(__name__)

  def cutFrames(self, videoPath: str, outputDir: str):
    """
    Extract frames from a video using OpenCV and save them to the output directory,
    preserving the original FPS (implicitly by grabbing every frame).
    """
    if not os.path.exists(videoPath):
      self.logger.error(f"Video file not found: {videoPath}")
      raise FileNotFoundError(f"Video file not found: {videoPath}")

    os.makedirs(outputDir, exist_ok=True)

    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
      self.logger.error(f"Failed to open video: {videoPath}")
      raise RuntimeError(f"Failed to open video: {videoPath}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    self.logger.info(f"Original FPS: {fps}, Total frames: {total_frames}")

    frame_idx = 0
    saved_frames = 0

    while True:
      ret, frame = cap.read()
      if not ret:
        break
      cv2.imwrite(os.path.join(outputDir, f"{frame_idx:06d}.png"), frame)
      frame_idx += 1
      saved_frames += 1

    cap.release()
    self.logger.info(f"Extracted {saved_frames} frames to {outputDir}")

  def clrMetadata(self):
    return None

  def getMetadata(self, videoPath: str):
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", videoPath
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    return json.loads(result.stdout)

  # Using FFmpeg to get the number of audio tracks in the file

  def countAudioTracks(self, videoPath: str) -> int:
    cmd_probe = [
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0",
        videoPath
    ]
    result = subprocess.run(
        cmd_probe, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    return int(result.stdout)

  def extractAudioTracks(self, videoPath: str, output_dir: str) -> list:
    os.makedirs(output_dir, exist_ok=True)

    # First get number of audio tracks
    cntAudioTracks = self.countAudioTracks(videoPath)

    output_files = []

    for idx in range(cntAudioTracks):
      out_path = os.path.join(output_dir, f"audio_track_{idx}.mp3")
      cmd_extract = [
          "ffmpeg", "-y", "-i", videoPath, "-map", f"0:a:{idx}", "-c:a", "mp3", out_path
      ]
      subprocess.run(cmd_extract, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
      output_files.append(out_path)

    return output_files

  def attachAudioTracks(self, videoPath: str, audio_tracks: list, output_path: str):
    if not audio_tracks:
      raise ValueError("No audio tracks provided.")

    if len(audio_tracks) != 1:
      self.logger.info("Processing with [multiple audio tracks]")
    else:
      self.logger.info("Processing with [single audio track]")

    cmd = ["ffmpeg", "-y", "-i", videoPath]

    # Add all audio inputs
    for track in audio_tracks:
      cmd.extend(["-i", track])

    # Map the video stream
    cmd.append("-map")
    cmd.append("0:v:0")

    # Map each audio stream
    for i in range(1, len(audio_tracks) + 1):
      cmd.extend(["-map", f"{i}:a:0"])

    # Copy codecs directly without re-encoding
    cmd.extend(["-c:v", "copy", "-c:a", "aac"])

    # Output file
    cmd.append(output_path)

    subprocess.run(cmd, check=True)

  def convertFrame2Time(self, frameIndex: int, fps: int) -> str:
    # Return in HH:MM:SS format
    return str(datetime.timedelta(seconds=(frameIndex / fps))).split('.')[0]

  # Get a specific frame from the input video
  #
  # @param videoPath: str - The path to the video file
  # @param frameIndex: int - The index of the frame to get
  def getFrame(self, videoPath: str, frameIndex: int = None):

    # Open video and get random frame
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
      raise RuntimeError("Could not open video file")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
    ret, frame = cap.read()

    cap.release()
    return frame
