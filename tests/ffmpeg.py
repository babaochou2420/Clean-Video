
from daos.MediaHelper import MediaHelper

metadata = MediaHelper().get_video_metadata("test.mp4")

print(metadata)
