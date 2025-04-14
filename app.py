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


class VideoProcessor:
  def __init__(self):
    self.temp_dir = "temp"
    os.makedirs(self.temp_dir, exist_ok=True)
    self.inpainter = VideoInpainter()

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

  def process_video(self, video_path, progress=gr.Progress()):
    """Main video processing pipeline"""
    try:
      # Process video with ProPainter
      output_path = "output_video.mp4"
      result = self.inpainter.process_video(video_path, output_path, progress)

      # Copy audio from original to processed video
      temp_video = "temp_video.mp4"
      cmd = [
          'ffmpeg', '-i', result,
          '-i', video_path,
          '-c:v', 'copy',
          '-c:a', 'aac',
          '-map', '0:v:0',
          '-map', '1:a:0',
          output_path
      ]
      subprocess.run(cmd)
      os.remove(temp_video)

      return output_path
    except Exception as e:
      return str(e)


def process_video_interface(video_path, progress=gr.Progress()):
  processor = VideoProcessor()
  try:
    output_path = processor.process_video(video_path)
    return output_path
  except Exception as e:
    return str(e)


def create_app():
  with gr.Blocks() as demo:
      # Create tabs
    home_tab = create_home_tab()
    settings_tab = create_settings_tab()
    about_tab = create_about_tab()

  return demo


if __name__ == "__main__":
  demo = create_app()
  demo.launch()
