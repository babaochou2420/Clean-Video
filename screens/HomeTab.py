import datetime
import logging
import gradio as gr
from daos.MediaHelper import MediaHelper
from utils.config import Config
from utils.logger import setup_logger
from video_inpainter import VideoInpainter
import cv2
from daos.enums.ModelEnum import ModelEnum, OpenCVFTEnum
import numpy as np


class HomeTab:

  def __init__(self):
    self.videoHelper = MediaHelper()
    self.config = Config.get_config()

  def __load__(self):

    logger = setup_logger('HomeTab')

    inpainter = VideoInpainter()

    stateVideoDuration = gr.State(value=0)
    stateVideoFrameTotal = gr.State(value=0)

    stateVideoFPS = gr.State(value=0)
    # Image: np.ndarray
    stateVideoFrame4Preview = gr.State(value=None)

    stateVideoMetadata = gr.State(value=None)

    with gr.Tab("Home") as tab:
      gr.Markdown("# Video Watermark & Subtitle Remover")

      with gr.Column():
        # Input Infomation
        with gr.Row(equal_height=True):
          with gr.Column(scale=4):
            input_video = gr.Video(label="Input Video")

          # Inpaint Result Preview
          with gr.Column(scale=6):
            with gr.Row():
              with gr.Column(scale=4):
                with gr.Column():
                  # Video Info
                  with gr.Row():
                    infoFPS = gr.Slider(minimum=1, maximum=60,
                                        value=30, step=1, label="FPS", interactive=False)
                    infoTotalFrames = gr.Number(
                        label="Total Frames", interactive=False)
                    infoVideoResolution = gr.Textbox(
                        label="Resolution", interactive=False)

                    infoVideoDuration = gr.Textbox(
                        label="Duration", interactive=False)
                  with gr.Row():

                    infoBitrate = gr.Textbox(
                        label="Bitrate", interactive=False)
                    infoAudioTracks = gr.Textbox(
                        label="Audio Tracks", interactive=False)

                  # def calFramesTotal(fps, duration):
                  #   return fps*duration

                  def update_video_info(video_path):
                    if not video_path:
                      return None, None, None, None
                    cap = cv2.VideoCapture(video_path)

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = total_frames // fps
                    durationFormatted = datetime.timedelta(seconds=duration)
                    cap.release()

                    metadata = self.videoHelper.getMetadata(video_path)

                    return metadata, durationFormatted, f"{metadata['streams'][0]['width']}x{metadata['streams'][0]['height']}", fps, duration, total_frames, fps, metadata["streams"][0]["nb_frames"]

                  input_video.change(
                      fn=update_video_info,
                      inputs=[input_video],
                      outputs=[stateVideoMetadata, infoVideoDuration, infoVideoResolution,
                               infoFPS, stateVideoDuration, stateVideoFrameTotal, stateVideoFPS, infoTotalFrames
                               ]
                  )

          def initVideoInfo():
            return None, None, None, None, None, None, None, None

          # input_video.clear()

        # Preview and Inpaint Settings
        with gr.Row(equal_height=True):
          with gr.Column(scale=4):
            preview_gallery = gr.Gallery(
                format="png",
                label="Preview",
                show_label=True,
                elem_id="preview_gallery",
                columns=2,
                height="auto",
                preview=True,
                allow_preview=True,
                object_fit="contain"
            )
          with gr.Column(scale=6):
            with gr.Row():
              # Preview Frame Control
              with gr.Column(scale=1):
                getPreviewFrame = gr.Button("Find Random Frame")

                nowFrameIndex = gr.Number(
                    label="Current Frame Index")
                nowFrameTime = gr.Textbox(
                    show_label=False, interactive=False)

                def showPreviewFrame(videoPath: str, frameIndex: int, fps: int):
                  frame = self.videoHelper.getFrame(videoPath, frameIndex)
                  return frame, [(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "Frame for Preview Generation")], self.videoHelper.convertFrame2Time(frameIndex, fps)

                nowFrameIndex.change(fn=showPreviewFrame, inputs=[
                    input_video, nowFrameIndex, stateVideoFPS], outputs=[stateVideoFrame4Preview, preview_gallery, nowFrameTime])

                def getRandomFrameIndex(frameTotal: int = 0):
                  if frameTotal == 0:
                    return gr.Info("Please input a video first")
                  else:
                    return np.random.randint(0, frameTotal)

                getPreviewFrame.click(fn=getRandomFrameIndex, inputs=[
                    stateVideoFrameTotal], outputs=[nowFrameIndex])

                genPreviewBtn = gr.Button("Generate Preview")

                def generate_preview(frame, model, enableFTransform, inpaintRadius):
                  try:

                    maskOverlay, processed = inpainter.genPreview(
                        frame, model, enableFTransform, inpaintRadius)

                    # Convert images to RGB for display
                    mask_rgb = cv2.cvtColor(maskOverlay, cv2.COLOR_BGR2RGB)
                    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

                    return [(mask_rgb, "Mask Overlay"), (processed_rgb, "Processed")]
                  except Exception as e:
                    logger.error(f"Error generating preview: {e}")
                    return str(e)

              # Inpaint Model Settings
              with gr.Column(scale=5):
                with gr.Row():
                  with gr.Column(variant="panel"):
                    gr.Markdown("# Inpaint", elem_classes=["center"])

                    modelPicker = gr.Radio(
                        choices=ModelEnum.values(),
                        label="Inpainting Model",
                        value=ModelEnum.OPENCV
                    )
                    inpaintRadiusSlider = gr.Slider(
                        minimum=1, maximum=9, value=3, step=1, label="Inpaint Radius")

                    with gr.Row():
                      with gr.Column(scale=1):
                        switchFTransform = gr.Checkbox(
                            visible=True, label="Enable FTransform", value=False)
                      with gr.Column(scale=7):
                        modeOpenCVFTInpaint = gr.Radio(
                            visible=False, choices=OpenCVFTEnum.values(), value=OpenCVFTEnum.ITERATIVE, label="Algorithm")

                    def showOpenCVFTAlgorithms(bool):
                      return gr.update(
                          visible=bool
                      )

                    switchFTransform.change(fn=showOpenCVFTAlgorithms, inputs=[
                                            switchFTransform], outputs=[modeOpenCVFTInpaint])

                    def showModelSettings_OpenCV(model):
                      if model == ModelEnum.OPENCV.value:
                        return gr.update(visible=True)
                      else:
                        return gr.update(visible=False)

                    modelPicker.change(fn=showModelSettings_OpenCV, inputs=[
                        modelPicker], outputs=[switchFTransform])

      # Output
      with gr.Row():
        with gr.Column(scale=4):
          output_video = gr.Video(label="Processed Video")
        with gr.Column(scale=6):
          with gr.Row(variant="panel"):
            with gr.Column(scale=1):
              switchKeepFormat = gr.Checkbox(
                  label="Remain Format", value=self.config["settings"]["FORMAT"]["SAME"])
              with gr.Row():
                formatVideo = gr.Dropdown(
                    choices=["mp4", "mkv", "avi", "mov", "webm"], value="mp4", label="Video Format")
                formatAudio = gr.Dropdown(
                    choices=["mp3", "aac", "m4a", "wav", "flac"], value="mp3", label="Audio Format")
            with gr.Column(scale=1):
              outputFolder = gr.Textbox(
                  label="Output Folder", value="output_video")
              outputFilename = gr.Textbox(
                  label="Output Filename", value="output_video")

          process_btn = gr.Button("Process Video")

      genPreviewBtn.click(
          fn=generate_preview,
          inputs=[stateVideoFrame4Preview, modelPicker,
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

  # def __widgetVideoInfo(self):
