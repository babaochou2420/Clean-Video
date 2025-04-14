import gradio as gr
from video_inpainter import InpaintMode


def create_settings_tab():
  with gr.Tab("Settings") as tab:
    gr.Markdown("## Processing Settings")

    with gr.Row():
      with gr.Column():
        model_choice = gr.Radio(
            choices=["STTN", "LAMA"],
            value="STTN",
            label="Inpainting Model",
            info="STTN: Better for videos, LAMA: Better for static images"
        )

        mask_height = gr.Slider(
            minimum=0.1,
            maximum=0.5,
            value=0.2,
            step=0.05,
            label="Mask Height (percentage of frame)"
        )

        device = gr.Radio(
            choices=["Auto", "CPU", "GPU"],
            value="Auto",
            label="Processing Device"
        )

        save_btn = gr.Button("Save Settings")

    def save_settings(model, height, device_choice):
      # TODO: Implement settings saving
      return f"Settings saved! Using {model} model"

    save_btn.click(
        fn=save_settings,
        inputs=[model_choice, mask_height, device],
        outputs=gr.Textbox(label="Status")
    )

  return tab
