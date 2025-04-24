
from daos.MediaHelper import MediaHelper


videoHelper = MediaHelper()

tracks = videoHelper.extractAudioTracks("test.mp4", "tests/audio")

print(tracks)

videoHelper.attachAudioTracks(
    "output.mp4", tracks, "tests/output_audio.mp4")
