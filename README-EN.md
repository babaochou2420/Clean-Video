# CleanVideo
[English](README-EN.md) | [繁體中文](README.md) 

A powerful video inpainting tool that helps remove unwanted elements from your videos while preserving the original quality and audio tracks.

[![Blog](https://img.shields.io/badge/Visit%20My%20Blog-babaochou2420.com-blue)](https://babaochou2420.com)

## Features

- **Multiple Inpainting Algorithms**:
  - OpenCV (FT) - Fast and efficient for simple removals
  - LaMa - Advanced AI-powered inpainting for complex scenes
- **Multi-track Audio Support**: Preserves all audio tracks from the original video
- **User-friendly Interface**: Simple and intuitive Gradio-based UI
- **Preview Capabilities**: See the results before final processing
- **Batch Processing**: Process multiple videos efficiently

## Installation

### Prerequisites

1. **Python 3.8+**
2. **FFmpeg**
   - Windows:
     ```bash
     # Using Chocolatey
     choco install ffmpeg
     
     # Or download from https://ffmpeg.org/download.html
     ```
   - Linux:
     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```
   - macOS:
     ```bash
     brew install ffmpeg
     ```

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/CleanVideo.git
   cd CleanVideo
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   gradio app.py
   ```

2. Open your browser and navigate to `http://127.0.0.1:15682`

3. Upload your video and select the inpainting algorithm

4. Process the video and download the result

## Limitations

- Processing time depends on total frames and algorithm complexity
- LaMa algorithm requires more computational resources
- Very complex scenes might require manual adjustments
- The running time on GPU might differ on your system

## System Requirements

- CPU: 4+ cores recommended
- RAM: 8GB minimum, 16GB recommended
- GPU: Tested with RTX 3050Ti
- Storage: 10GB+ free space for processing

## Contributing

Although it's a project now stopped, but with your supports will give me more motivation to make it better and stronger instead of a self-usage tool. 
