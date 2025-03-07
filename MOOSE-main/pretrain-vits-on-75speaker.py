from transformers import AutoModel, Wav2Vec2Processor, VivitModel, VivitImageProcessor, VivitConfig, AutoImageProcessor, TimesformerModel
from tqdm import tqdm
import cv2
import argparse
import time
import numpy as np
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch.nn as nn
import glob
from tqdm import tqdm
def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)

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
    # num_frames = len(frames)
    # print("num_frames original" , num_frames)
    return frames

def get_audio_target(audio_path, audio_model, audio_processor):
    # load dummy dataset and read soundfiles
    y, sr = librosa.load(audio_path, sr = 16000)
    rtmri, _ = librosa.effects.trim(y)

    # tokenize
    input_values = audio_processor(y, sampling_rate=16000, return_tensors="pt").input_values

    # retrieve logits
    with torch.no_grad():
        logits = audio_model(input_values).logits


    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)

    return predicted_ids

# load model and processor
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")


# video_paths = glob.glob("/data1/hongn/chunks/video/*.avi")
# audio_paths = ["/data1/hongn/chunks/audio/" + i.split('/')[-1].split('.')[0] + ".wav" for i in video_paths]

# video_paths = glob.glob("/data1/span_data/rtmri75s/sub*/2drt/video/*.mp4")
# audio_paths = ["/data1/span_data/rtmri75s/" + i.split('/')[-4] + "/2drt/audio/" + i.split('/')[-1].split('.')[0][:-5] + "audio.wav" for i in video_paths]

all_spk = ['sub002',
'sub003',
'sub004',
'sub005',
'sub007',
'sub008',
'sub009',
'sub010',
'sub011',
'sub013',
'sub014',
'sub015',
'sub016',
'sub017',
'sub018',
'sub019',
'sub020',
'sub023',
'sub025',
'sub027',
'sub031',
'sub036',
'sub040',
'sub042',
'sub043',
'sub044',
'sub045',
'sub046',
'sub047',
'sub048',
'sub049',
'sub050',
'sub054',
'sub055',
'sub057',
'sub059',
'sub067',
'sub070',
'sub071',
'sub072']

video_paths = []
for sub in all_spk:
    video_paths = video_paths + glob.glob(f"/data1/span_data/rtmri75s/{sub}/2drt/video/*.mp4")
audio_paths = ["/data1/span_data/rtmri75s/" + i.split('/')[-4] + "/2drt/audio/" + i.split('/')[-1].split('.')[0][:-5] + "audio.wav" for i in video_paths]

BATCH = 4
FPS = 86*BATCH
AUDIO_UNIT = 50*BATCH

device = 'cuda:2'
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.classifier = nn.Linear(in_features = 768, out_features = 392)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.9, step_size=5)
best_loss = float('inf')
criterion = nn.CTCLoss(blank=0)

EPOCH = 100
model = model.to(device)
print("start traning with epoch 0")
for e in range(EPOCH):
    epoch_loss = 0
    for i in tqdm(range(len(video_paths))):

        video_path = video_paths[i]
        audio_path = audio_paths[i]
        frames = get_frames(video_path)
        input_len = int(len(frames)/FPS)
        audio_target = get_audio_target(audio_path, audio_model, audio_processor).to(device)
        # print(frames.shape, input_len)
        # assert False
        for j in range(input_len):
            optimizer.zero_grad()

            inputs = image_processor(images=frames[j*FPS:(j+1)*FPS], return_tensors="pt").to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            predicted_ids = audio_target[:,j*AUDIO_UNIT:(j+1)*AUDIO_UNIT]

            loss = criterion(log_probs.unsqueeze(1), predicted_ids, input_lengths=tuple([1]), target_lengths=tuple([1]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            # print(loss)
            epoch_loss += loss.item()

    print(f"epoch {e} loss: ", epoch_loss, f"Begin training with epoch {e+1}")
    torch.save(model, "/data1/hongn/chunks/ckpt_pretrained_vits_on_span/pretrained_vit_on_75engspeaker_epoch{e}.pt")

    if(epoch_loss < best_loss):
        best_loss = epoch_loss
        torch.save(model, "/data1/hongn/chunks/ckpt_pretrained_vits_on_span/best_eng.pt")