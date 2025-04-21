
from daos.VideoHelper import VideoHelper


videoHelper = VideoHelper()

tracks = videoHelper.extractAudioTracks("test.mp4", "tests/audio")

print(tracks)

videoHelper.attachAudioTracks(
    "output.mp4", tracks, "tests/output_audio.mp4")
