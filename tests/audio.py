
from daos.VideoHelper import VideoHelper


videoHelper = VideoHelper()

tracks = videoHelper.audioTracksExtract("test.mp4", "tests/audio")

print(tracks)

videoHelper.audioTracksReattach(
    "output.mp4", tracks, "tests/output_audio.mp4")
