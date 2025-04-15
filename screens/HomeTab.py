import datetime
import gradio as gr
from video_inpainter import VideoInpainter

import cv2
from daos.enums.ModelEnum import ModelEnum


class HomeTab:
  def __load__():

    stateVideoDuration = gr.State(value=0)

    with gr.Tab("Home") as tab:
      gr.Markdown("# Video Watermark & Subtitle Remover")

      with gr.Row():
        with gr.Column():
          input_video = gr.Video(label="Input Video")
          with gr.Row():
            infoFPS = gr.Slider(minimum=1, maximum=60,
                                value=30, step=1, label="FPS", interactive=True)
            infoTotalFrames = gr.Number(
                label="Total Frames", interactive=False)
            infoVideoResolution = gr.Textbox(
                label="Resolution", interactive=False)

          infoVideoLength = gr.Textbox(
              label="Duration", interactive=False)
          video_state = gr.State({
              'fps': None,
              'width': None,
              'height': None,
              'total_frames': None,
              'length_ms': None
          })

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

        with gr.Column():
          output_video = gr.Video(label="Processed Video")

      with gr.Row():
        with gr.Column(variant="panel"):
          # gr.Markdown("# Inpaint", elem_classes=["center"])
          switchFTransform = gr.Checkbox(
              label="Enable FTransform", value=False)
        with gr.Row():
          modelPicker = gr.Radio(
              choices=ModelEnum.values(),
              label="Inpainting Model",
              value=ModelEnum.LAMA
          )

      def process_video(video_path, model, enableFTransform, progress=gr.Progress(track_tqdm=True)):
        try:
          inpainter = VideoInpainter()
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
