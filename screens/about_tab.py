import gradio as gr


def create_about_tab():
  with gr.Tab("About") as tab:
    gr.Markdown("""
        # About Video Watermark & Subtitle Remover
        
        This application uses ProPainter to remove watermarks and subtitles from videos.
        
        ## Features
        - Automatic watermark/subtitle removal
        - GPU acceleration support
        - Customizable processing settings
        
        ## Technology
        - PyTorch: Deep learning framework
        - Gradio: Web interface
        
        ## License
        MIT License
        """)

  return tab
