import gradio as gr
import cv2
import numpy as np
import torch
import os
from tqdm import tqdm
import subprocess
from datetime import timedelta
from video_inpainter import VideoInpainter
from screens.HomeTab import HomeTab
from screens.settings_tab import create_settings_tab
from screens.about_tab import create_about_tab
from utils.logger import setup_logger, log_function
from typing import Optional, Tuple, Union

# Setup logger
logger = setup_logger('app')


@log_function(logger)
def create_app():
  logger.info("Creating Gradio interface")
  with gr.Blocks() as demo:
      # Create tabs
    logger.debug("Creating tabs")
    HomeTab.__load__()
    settings_tab = create_settings_tab()
    about_tab = create_about_tab()
    logger.info("Interface created successfully")

  return demo


if __name__ == "__main__":
  logger.info("Starting application")
  demo = create_app()
  logger.info("Launching Gradio interface")
  demo.launch()
