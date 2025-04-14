import os
import subprocess
import tempfile
from typing import List, Dict
import logging


class VideoHelper:
  def __init__(self):
    self.audio_cache = {}
    self.temp_dir = tempfile.gettempdir()
    self.logger = logging.getLogger(__name__)

  def audioDetach(self, filepath: str) -> Dict[str, str]:
    """
    Extract audio tracks from video file and store them temporarily.
    Returns a dictionary mapping track indices to temporary audio file paths.
    """
    if not os.path.exists(filepath):
      self.logger.error(f"File not found: {filepath}")
      return {}

    # Get audio track information
    cmd = ['ffprobe', '-v', 'quiet', '-print_format',
           'json', '-show_streams', filepath]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
      self.logger.error("Failed to get stream information")
      return {}

    # Parse streams to find audio tracks
    import json
    streams = json.loads(result.stdout)['streams']
    audio_tracks = [s for s in streams if s['codec_type'] == 'audio']

    if not audio_tracks:
      self.logger.warning("No audio tracks found in the video")
      return {}

    # Extract each audio track
    self.audio_cache = {}
    for i, track in enumerate(audio_tracks):
      temp_audio = os.path.join(self.temp_dir, f"temp_audio_{i}.m4a")
      cmd = [
          'ffmpeg', '-i', filepath,
          '-map', f'0:{track["index"]}',
          '-c:a', 'copy',
          '-y', temp_audio
      ]

      result = subprocess.run(cmd, capture_output=True)
      if result.returncode == 0:
        self.audio_cache[str(i)] = temp_audio
        self.logger.info(f"Extracted audio track {i} to {temp_audio}")
      else:
        self.logger.error(f"Failed to extract audio track {i}")

    return self.audio_cache

  def videoConstruct(self, frames_dir: str, output_path: str, fps: float) -> bool:
    """
    Construct video from frames and combine with cached audio tracks.
    Returns True if successful, False otherwise.
    """
    if not self.audio_cache:
      self.logger.warning("No audio tracks found in cache")
      return False

    # Create temporary video without audio
    temp_video = os.path.join(self.temp_dir, "temp_video.mp4")
    cmd = [
        'ffmpeg', '-framerate', str(fps),
        '-i', os.path.join(frames_dir, '%d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-y', temp_video
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
      self.logger.error("Failed to create temporary video")
      return False

    # Combine video with audio tracks
    cmd = ['ffmpeg', '-i', temp_video]
    for track_path in self.audio_cache.values():
      cmd.extend(['-i', track_path])

    cmd.extend(['-c:v', 'copy'])
    for i in range(len(self.audio_cache)):
      cmd.extend(['-c:a:' + str(i), 'copy'])

    cmd.extend(['-y', output_path])

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
      self.logger.error("Failed to combine video with audio")
      return False

    # Clean up temporary files
    try:
      os.remove(temp_video)
      for audio_file in self.audio_cache.values():
        os.remove(audio_file)
      self.audio_cache.clear()
    except Exception as e:
      self.logger.warning(f"Failed to clean up temporary files: {e}")

    return True

  def __del__(self):
    """Clean up any remaining temporary files when the object is destroyed"""
    try:
      for audio_file in self.audio_cache.values():
        if os.path.exists(audio_file):
          os.remove(audio_file)
    except Exception as e:
      self.logger.warning(
          f"Failed to clean up temporary files in destructor: {e}")
