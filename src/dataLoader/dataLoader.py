import librosa
import numpy as np
import os
import pretty_midi
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch


def load_audio(file_path, sr=22050):
    """Load an audio file and return the time series and sampling rate."""
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr


def extract_mel_spectrogram(y, sr, n_mels=128, hop_length=512):
    """Extract Mel spectrogram from audio time series."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


def load_midi(file_path):
    """
    Load a MIDI file and return structured info about notes.
    Returns a list of dictionaries: {pitch, start, end, velocity}
    """
    if not os.path.exists(file_path):
        print(f"[WARNING] MIDI file not found: {file_path}")
        return None

    try:
        midi = pretty_midi.PrettyMIDI(file_path)

        notes = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue  # ignore drums
            for n in instrument.notes:
                notes.append({
                    "pitch": n.pitch,
                    "start": n.start,
                    "end": n.end,
                    "velocity": n.velocity
                })

        return notes

    except Exception as e:
        print(f"[ERROR] Could not load MIDI {file_path}: {e}")
        return None


def load_dataset(data_dir):
    """Load all audio files in a directory and extract their Mel spectrograms."""
    dataset = {}

    for filename in os.listdir(data_dir):
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            audio_path = os.path.join(data_dir, filename)

            # Compute mel spectrogram
            y, sr = load_audio(audio_path)
            mel_spectrogram = extract_mel_spectrogram(y, sr)

            # Try to load MIDI with same base name
            midi_path = os.path.splitext(audio_path)[0] + ".mid"
            midi = load_midi(midi_path)

            dataset[filename] = (mel_spectrogram, midi)

    return dataset


class NSynthDataset(Dataset):
    def __init__(self, tfrecord_path, wav_dir, index_path=None, mel_cache_dir="cache/mels"):
        self.wav_dir = wav_dir
        self.mel_cache_dir = mel_cache_dir
        # Définition du schéma TFRecord
        description = {
            "pitch": "int",
            "velocity": "int",
            "instrument": "int",
            "audio": "float",
            "note_str": "byte"
        }
        self.tfrecord_data = TFRecordDataset(tfrecord_path, index_path, description)
        # Calculer la longueur à partir du fichier d'index si fourni
        self._length = None
        if index_path is not None and os.path.exists(index_path):
            with open(index_path, 'r') as f:
                self._length = sum(1 for _ in f)

        # Charger tous les noms de fichiers et labels (pitch)
        self.samples = []
        print("Chargement des chemins de spectrogrammes et labels...")
        missing = 0
        missing_files = []
        found = 0
        for i, sample in enumerate(tqdm(self.tfrecord_data, total=self._length)):
            file_id = sample["note_str"].decode("utf-8")
            mel_path = os.path.join(self.mel_cache_dir, file_id + ".npy")
            pitch = sample["pitch"]
            if os.path.exists(mel_path):
                self.samples.append((mel_path, pitch))
                found += 1
            else:
                missing += 1
                missing_files.append(mel_path)
        print(f"[DEBUG] {found} fichiers de spectrogrammes trouvés dans le cache.")
        if missing > 0:
            print(f"[WARNING] {missing} fichiers de spectrogrammes manquants dans le cache. Exemples:")
            for m in missing_files[:10]:
                print(f"  - {m}")
            if missing > 10:
                print(f"  ...et {missing-10} autres non affichés.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel_path, pitch = self.samples[idx]
        mel = np.load(mel_path)
        mel = np.expand_dims(mel, axis=0)  # (1, n_mels, time)
        mel = torch.tensor(mel, dtype=torch.float32)
        pitch = torch.tensor(pitch, dtype=torch.long)
        if pitch.dim() > 0:
            pitch = pitch.squeeze()
        return mel, pitch


# -------------------------
# MAIN EXECUTION EXAMPLE
# -------------------------
if __name__ == "__main__":

    #from tfrecord.tools import tfrecord2idx

    #tfrecord_path = "nsynth-valid.tfrecord"
    #index_path = "nsynth-valid.idx"

    #tfrecord2idx.create_index(tfrecord_path, index_path)

    import matplotlib.pyplot as plt

    data_dir = "./src/dataLoader/"  

    print("Chargement du dataset…")
    dataset = load_dataset(data_dir)

    print(f"Nombre de fichiers trouvés : {len(dataset)}")

    dataset = NSynthDataset(
        tfrecord_path="nsynth-valid.tfrecord",
        wav_dir="nsynth-valid/audio",
        index_path="nsynth-valid.idx"
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Example: inspect one file
    for filename, (mel, midi) in dataset.items():
        print(f"\n=== {filename} ===")
        print(f"Mel spectrogram shape : {mel.shape}")

        if midi:
            print(f"{len(midi)} notes trouvées dans le MIDI")
            print("Exemple note :", midi[0])
        else:
            print("Pas de fichier MIDI correspondant.")

        # Afficher le spectrogramme du premier fichier
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=22050)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel Spectrogram: {filename}")
        plt.tight_layout()
        plt.show()

        break  # remove if you want to process all files

    