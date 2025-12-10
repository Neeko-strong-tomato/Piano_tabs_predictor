import librosa
import numpy as np
import os
import pretty_midi


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


# -------------------------
# MAIN EXECUTION EXAMPLE
# -------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    data_dir = "./src/dataLoader/"  

    print("Chargement du dataset…")
    dataset = load_dataset(data_dir)

    print(f"Nombre de fichiers trouvés : {len(dataset)}")

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
