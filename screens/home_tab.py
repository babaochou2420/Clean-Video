import gradio as gr
from video_inpainter import VideoInpainter


def create_home_tab():
  with gr.Tab("Home") as tab:
    gr.Markdown("# Video Watermark & Subtitle Remover")
    gr.Markdown(
        "Upload a video to remove watermarks or subtitles using ProPainter")

    with gr.Row():
      with gr.Column():
        input_video = gr.Video(label="Input Video")
        process_btn = gr.Button("Process Video")

      with gr.Column():
        output_video = gr.Video(label="Processed Video")

    def process_video(video_path, progress=gr.Progress(track_tqdm=True)):
      try:
        inpainter = VideoInpainter()
        output_path = "output_video.mp4"
        result = inpainter.process_video(
            video_path, output_path, 0.2, progress=progress)
        return result
      except Exception as e:
        return str(e)

    process_btn.click(
        fn=process_video,
        inputs=[input_video],
        outputs=output_video
    )

  return tab
