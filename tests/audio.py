
from daos.VideoHelper import VideoHelper


videoHelper = VideoHelper()

tracks = videoHelper.extract_all_audio_tracks_ffmpeg("test.mp4", "tests/audio")

print(tracks)

videoHelper.reattach_audio_tracks_ffmpeg(
    "output.mp4", tracks, "tests/output_audio.mp4")
