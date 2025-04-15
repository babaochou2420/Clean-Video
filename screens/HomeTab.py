import datetime
import logging
import gradio as gr
from utils.logger import setup_logger
from video_inpainter import VideoInpainter
import cv2
from daos.enums.ModelEnum import ModelEnum
import numpy as np


class HomeTab:

  def __load__():

    logger = setup_logger('HomeTab')

    inpainter = VideoInpainter()

    stateVideoDuration = gr.State(value=0)

    with gr.Tab("Home") as tab:
      gr.Markdown("# Video Watermark & Subtitle Remover")

      # Info & Settings
      with gr.Column():
        with gr.Column():
          with gr.Row():
            input_video = gr.Video(label="Input Video")

            # Inpaint Result Preview
            with gr.Row():
              with gr.Column():
                preview_gallery = gr.Gallery(
                    label="Preview",
                    show_label=True,
                    elem_id="preview_gallery",
                    columns=2,
                    height="auto",
                    preview=True,
                    allow_preview=True
                )
                genPreviewBtn = gr.Button("Generate Preview")

            def generate_preview(video_path, model, enableFTransform, inpaintRadius):
              if not video_path:
                return None
              try:

                maskOverlay, processed = inpainter.genPreview(
                    video_path, model, enableFTransform, inpaintRadius)

                # Convert images to RGB for display
                mask_rgb = cv2.cvtColor(maskOverlay, cv2.COLOR_BGR2RGB)
                processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

                return [(mask_rgb, "Mask Overlay"), (processed_rgb, "Processed")]
              except Exception as e:
                logger.error(f"Error generating preview: {e}")
                return str(e)
        with gr.Row():
          with gr.Column():
            # Video Info
            with gr.Row():
              infoFPS = gr.Slider(minimum=1, maximum=60,
                                  value=30, step=1, label="FPS", interactive=True)
              infoTotalFrames = gr.Number(
                  label="Total Frames", interactive=False)
              infoVideoResolution = gr.Textbox(
                  label="Resolution", interactive=False)

            infoVideoLength = gr.Textbox(
                label="Duration", interactive=False)

            def calFramesTotal(fps, duration):
              return fps*duration

            def update_video_info(video_path):
              if not video_path:
                return None, None, None, None
              cap = cv2.VideoCapture(video_path)
              width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
              height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
              total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
              fps = cap.get(cv2.CAP_PROP_FPS)
              duration = total_frames // fps
              durationFormatted = datetime.timedelta(seconds=duration)
              cap.release()

              return durationFormatted, f"{width}x{height}", fps, duration

            input_video.change(
                fn=update_video_info,
                inputs=[input_video],
                outputs=[infoVideoLength, infoVideoResolution,
                         infoFPS, stateVideoDuration]
            )

            infoFPS.change(fn=calFramesTotal, inputs=[
                infoFPS, stateVideoDuration], outputs=[infoTotalFrames])

            process_btn = gr.Button("Process Video")

          # Inpaint Model Settings
          with gr.Column():
            with gr.Row():
              with gr.Column(variant="panel"):
                gr.Markdown("# Inpaint", elem_classes=["center"])
                switchFTransform = gr.Checkbox(
                    label="Enable FTransform", value=False)
                modelPicker = gr.Radio(
                    choices=ModelEnum.values(),
                    label="Inpainting Model",
                    value=ModelEnum.OPENCV
                )
                inpaintRadiusSlider = gr.Slider(
                    minimum=1, maximum=9, value=3, step=1, label="Inpaint Radius")

      with gr.Row():
        output_video = gr.Video(label="Processed Video")

      genPreviewBtn.click(
          fn=generate_preview,
          inputs=[input_video, modelPicker,
                  switchFTransform, inpaintRadiusSlider],
          outputs=preview_gallery
      )

      def process_video(video_path, model, enableFTransform, progress=gr.Progress(track_tqdm=True)):
        try:
          output_path = "output_video.mp4"
          result = inpainter.process_video(
              video_path, output_path, model, enableFTransform)
          return result
        except Exception as e:
          return str(e)

      process_btn.click(
          fn=process_video,
          inputs=[input_video, modelPicker, switchFTransform],
          outputs=output_video
      )

    return tab
