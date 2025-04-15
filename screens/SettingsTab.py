import gradio as gr
from video_inpainter import InpaintMode


class SettingsTab:
  def __init__(self):
    None

  def __load__():
    with gr.Tab("Settings") as tab:
      gr.Markdown("## Processing Settings")

      with gr.Row():
        with gr.Column():
          None

      # def save_settings(model, height, device_choice):
      #   # TODO: Implement settings saving
      #   return f"Settings saved! Using {model} model"

      # save_btn.click(
      #     fn=save_settings,
      #     inputs=[model_choice, mask_height, device],
      #     outputs=gr.Textbox(label="Status")
      # )

    return tab
