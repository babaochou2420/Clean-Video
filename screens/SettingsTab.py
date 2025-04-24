import gradio as gr


class SettingsTab:
  def __init__(self):
    None

  def __load__(self):
    with gr.Tab("Settings") as tab:
      gr.Markdown("## Processing Settings")

      with gr.Row():
        with gr.Column():
          gr.Markdown("### Video Output Settings")
          output_format = gr.Dropdown(
              choices=["MP4", "MKV"],
              value="MP4",
              label="Output Format",
              info="Only MKV supports multiple audio tracks"
          )

    return tab
