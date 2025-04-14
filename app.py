import gradio as gr
import cv2
import numpy as np
import torch
import os
from tqdm import tqdm
import subprocess
from datetime import timedelta
from video_inpainter import VideoInpainter
from screens.home_tab import create_home_tab
from screens.settings_tab import create_settings_tab
from screens.about_tab import create_about_tab
from utils.logger import setup_logger, log_function
from typing import Optional, Tuple, Union

# Setup logger
logger = setup_logger('app')


class VideoProcessor:
  def __init__(self):
    self.logger = logger
    self.temp_dir = "temp"
    os.makedirs(self.temp_dir, exist_ok=True)
    self.inpainter = VideoInpainter()
    self.logger.info("VideoProcessor initialized")

  def extract_frames(self, video_path, frame_rate, progress=gr.Progress()):
    """Extract frames from video at specified frame rate"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    frames = []
    frame_indices = []
    current_frame = 0

    progress(0, desc="Extracting frames...")
    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      if current_frame % int(fps / frame_rate) == 0:
        frames.append(frame)
        frame_indices.append(current_frame)

        # Calculate current timestamp
        current_time = timedelta(seconds=current_frame/fps)
        total_time = timedelta(seconds=duration)
        progress(current_frame/total_frames,
                 desc=f"Processing frame at {current_time} / {total_time}")

      current_frame += 1

    cap.release()
    return frames, frame_indices, fps

  @log_function(logger)
  def process_video(self, video_path: str, height_percent: Union[float, str], progress: Optional[gr.Progress] = None) -> Tuple[Optional[str], str]:
    """Main video processing pipeline"""
    try:
      # Convert height_percent to float if it's a string
      if isinstance(height_percent, str):
        try:
          height_percent = float(height_percent)
        except ValueError:
          self.logger.error(f"Invalid height_percent value: {height_percent}")
          return None, f"Error: Invalid height percentage value: {height_percent}"

      self.logger.info(f"Starting video processing: {video_path}")
      self.logger.debug(f"Height percent: {height_percent}")

      # Create output directory if it doesn't exist
      os.makedirs("output", exist_ok=True)
      self.logger.debug("Output directory created/verified")

      # Generate output path
      output_path = os.path.join("output", "processed_video.mp4")
      self.logger.debug(f"Output path: {output_path}")

      # Process video
      self.logger.info("Starting video inpainting")
      result = self.inpainter.process_video(
          video_path, output_path, height_percent, progress)

      if result is None:
        self.logger.error("Video processing failed")
        return None, "Error processing video. Check debug logs for details."

      self.logger.info("Video processing completed successfully")
      return result, "Processing completed successfully"

    except Exception as e:
      self.logger.error(
          f"Error during video processing: {str(e)}", exc_info=True)
      return None, f"Error: {str(e)}"


def process_video_interface(video_path, height_percent, progress=gr.Progress()):
  processor = VideoProcessor()
  try:
    output_path = processor.process_video(video_path, height_percent)
    return output_path
  except Exception as e:
    return str(e)


@log_function(logger)
def create_app():
  logger.info("Creating Gradio interface")
  with gr.Blocks() as demo:
      # Create tabs
    logger.debug("Creating tabs")
    home_tab = create_home_tab()
    settings_tab = create_settings_tab()
    about_tab = create_about_tab()
    logger.info("Interface created successfully")

  return demo


if __name__ == "__main__":
  logger.info("Starting application")
  demo = create_app()
  logger.info("Launching Gradio interface")
  demo.launch()
