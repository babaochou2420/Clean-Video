# CleanVideo
[English](README.md) | [繁體中文](README.zh-TW.md) 

一個強大的影片修復工具，可以幫助您移除影片中不需要的元素，同時保持原始畫質和音軌。

[![部落格](https://img.shields.io/badge/參觀我的部落格-babaochou2420.com-blue)](https://babaochou2420.com)

## 功能特點

- **多種修復演算法**：
  - OpenCV (FT) - 快速且高效，適合簡單的移除
  - LaMa - 先進的 AI 驅動修復，適合複雜場景
- **提供多音軌支援**：保留原始影片的所有音軌（尚未完成）
- **使用者友善介面**：簡單直觀的 Gradio 介面
- **事前預覽功能**：在開始處理前針對指定畫面測試結果

## 安裝說明

### 必要條件

1. **Python 3.10+**
2. **FFmpeg**
   - Windows：
     ```bash
     # 使用 Chocolatey
     choco install ffmpeg
     
     # 或從 https://ffmpeg.org/download.html 下載
     ```
   - Linux：
     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```
   - macOS：
     ```bash
     brew install ffmpeg
     ```

### 設定步驟

1. 複製專案：
   ```bash
   git clone https://github.com/yourusername/CleanVideo.git
   cd CleanVideo
   ```

2. 建立並啟動虛擬環境：
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

3. 安裝依賴套件：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方式

1. 啟動應用程式：
   ```bash
   gradio app.py
   ```

2. 在瀏覽器中開啟 `http://127.0.0.1:15682`

3. 上傳影片並選擇修復演算法

4. 處理影片並下載結果

## 限制


- 目前並不支援手動遮罩繪製
- 處理時間取決於總幀數和演算法
  - 運行時間恐因設備能力而異

## 補充
- 由於 Pro-Painter 所需之 GPU 需求過高暫時不打算導入

## 系統需求

- CPU：建議 4+ 核心
- RAM：最少 8GB，建議 16GB
- GPU：測試使用 RTX 3050Ti
- 儲存空間：處理時需要 10GB+ 可用空間

## 貢獻

雖然這是一個目前暫停的專案，但有了您的支持，將能給我更多動力來改進和強化它，而不僅僅是一個自用工具。

