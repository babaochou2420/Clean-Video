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
from screens.SettingsTab import SettingsTab
# from screens.about_tab import create_about_tab
from utils.logger import setup_logger, log_function
from typing import Optional, Tuple, Union

# import os
# # ValueError: When localhost is not accessible, a shareable link must be created. Please set share=True or check your proxy settings to allow access to localhost.
# os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"

# Setup logger
logger = setup_logger('app')


@log_function(logger)
def create_app():
  # logger.info("Creating Gradio interface")
  with gr.Blocks() as demo:
      # Create tabs
    logger.debug("Creating tabs")
    HomeTab.__load__()
    SettingsTab.__load__()
    # about_tab = create_about_tab()
    # logger.info("Interface created successfully")

  return demo


# logger.info("Starting application")
demo = create_app()

# logger.info("Launching Gradio interface")
demo.launch(server_port=15682)
