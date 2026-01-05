import os
import numpy as np
import librosa
from tqdm import tqdm
import json

AUDIO_DIR = "nsynth-valid/audio"
CACHE_DIR = "cache/mels"
SR = 22050
N_MELS = 128
HOP_LENGTH = 512
EXAMPLES_JSON = "nsynth-valid/examples.json"

os.makedirs(CACHE_DIR, exist_ok=True)

# Lire la liste des fichiers attendus depuis le json
with open(EXAMPLES_JSON, "r") as f:
    examples = json.load(f)

file_ids = list(examples.keys())
missing = []
for file_id in tqdm(file_ids, desc="Precomputing Mel spectrograms"):
    wav_name = file_id + ".wav"
    audio_path = os.path.join(AUDIO_DIR, wav_name)
    cache_path = os.path.join(CACHE_DIR, file_id + ".npy")
    if not os.path.exists(audio_path):
        missing.append(wav_name)
        continue
    if os.path.exists(cache_path):
        continue
    try:
        y, sr = librosa.load(audio_path, sr=SR)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        S_dB = librosa.power_to_db(S, ref=np.max)
        np.save(cache_path, S_dB)
    except Exception as e:
        print(f"[ERROR] {wav_name}: {e}")
        missing.append(wav_name)

if missing:
    print("Missing or failed files:")
    for m in missing:
        print(m)
else:
    print("All files processed successfully.")