from transformers import AutoModel, Wav2Vec2Processor, VivitModel, VivitImageProcessor, VivitConfig, AutoImageProcessor, TimesformerModel
from tqdm import tqdm
import cv2
import argparse
import time
import numpy as np
import torch
import librosa

## SETTINGS
HOP_LENGTH = 320
NUM_FRAMES = 4
FRAME_RATE = 100
TIME_STEPS = 0.02 # seconds
FRAME_STEPS = int(TIME_STEPS * FRAME_RATE) # Frame per seconds, should be integer
device = 'cuda:2'
BATCH_SIZE = 8

# READ AUDIO AND VIDEO FILES
# y, sr = librosa.load("/data2/hongn/TimeSformer/usc_s1_2_0.wav", sr = 16000)
# rtmri, _ = librosa.effects.trim(y)
# Read all frames from the video
cap = cv2.VideoCapture("/data2/hongn/TimeSformer/usc_s1_2_0.avi")

frames = []
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
while True:
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (64, 64))
    frames.append(frame)
cap.release()
frames = np.array(frames, dtype=np.float32)  # Ensure float32 type
num_frames = len(frames)
print("num_frames original" , num_frames)

## UPSAMPLE by DUPPLICATE from 99 FPS to 100 FPS
missing_frames_from_framerate = int(duration*100) - num_frames
count = 0
temp = list(frames)
for i in list(range(0, num_frames, int(num_frames / missing_frames_from_framerate)))[:missing_frames_from_framerate]:
    temp.insert(i+count, frames[i+count])
    count += 1
frames = temp
print("num_frames inpterpolate framerate 100:" , len(frames))

## PUT IMAGES INTO CHUNKS OF 8-Frames VIDEOS (40 ms) with 20ms overlap
video_all = [] # Dimension (B, T, C, W, H)
for i in range(NUM_FRAMES): # Append NUM_FRAMES to the end of video
    frames.append(frames[-1])

frame_tensor = image_processor(list(frames), return_tensors="pt").pixel_values[0]
for i in range(0, len(frames) - NUM_FRAMES, FRAME_STEPS):
    video_all.append(frame_tensor[i:i + NUM_FRAMES])

video_all = torch.stack(video_all).to(device)

# PUTS SHORT VIDEOS INTO BATCHS
features = []
model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400", output_hidden_states=True, ignore_mismatched_sizes=True
        ).to(device)

for i in range(0, len(video_all), BATCH_SIZE):
        video_batch = video_all[i:i + BATCH_SIZE] # Get a batch of up to `batch_size` frames
    # If fewer than batch_size frames, pad with black frames (zeros)
        if len(video_batch) < BATCH_SIZE:
                pad_frames = BATCH_SIZE - len(video_batch)
                padding = torch.zeros((pad_frames, 8, 3, 224, 224), dtype=torch.float32).to(device)
                video_batch = torch.concat([video_batch, padding], axis=1)

        with torch.no_grad():
                video_outputs = model(pixel_values=video_batch)
                batch_features = video_outputs.last_hidden_state[:,0,:]  # Shape: [B, seq_len, feature_dim]
                features.append(batch_features)

## FINAL VIDEO FEATURES - sometime, the length video features might not match audio features because of frames-audio mismatch. Just dicard unmatch length
features = torch.concat(features , axis=0)

print(features.shape)

