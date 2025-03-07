import glob
import os
# import pandas as pd
import cv2 

folders = glob.glob("/data1/span_data/rtmri75s/sub0[0-7]*")
video_paths = []
audio_paths = []
for i in folders:
    sub_video_paths = glob.glob(i + "/2drt/video/*")
    video_paths = video_paths + sub_video_paths

def video_len_with_opencv(filename):
    video = cv2.VideoCapture(filename)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count/fps
    return duration, frame_count, fps

for vidpath in video_paths:
    i = 0
    duration, frame_count, fps = video_len_with_opencv(vidpath)
    while i < duration:
        chunk_name = '/data1/hongn/rtmri75s_processed/video/' + vidpath.split('/')[-1].split('.')[0] + '_chunk_{:.1f}.mp4'.format(i)
        os.system(f"ffmpeg -y -i {vidpath} -ss {i} -t {i + 0.2} {chunk_name}")
        i += 0.2