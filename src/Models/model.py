import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import librosa
# Simuler un audio mono de 5 secondes
sr = 16000
t = 5
y = np.sin(2*np.pi*440*np.linspace(0, t, sr*t))  # La note A4

# Calculer un spectrogramme (CQT ou Mel)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=256)
S_db = librosa.power_to_db(S, ref=np.max)

# Convertir en tenseur PyTorch et ajouter batch + channel
x = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, freq, time)
print("Input shape:", x.shape)

class SimpleCNN(nn.Module):
    def __init__(
        self,
        mode="notes",
        num_notes=127,
        num_families=10,
        num_sources=3
    ):
        super().__init__()
        assert mode in ["notes", "instrument"]
        self.mode = mode

        # ===== Backbone CNN =====
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # ===== Heads =====
        if self.mode == "notes":
            # Global pooling → Conv head
            self.final_conv = nn.Conv2d(256, num_notes, kernel_size=1)

        elif self.mode == "instrument":
            # Global pooling → MLP heads
            self.family_head = nn.Linear(256, num_families)
            self.source_head = nn.Linear(256, num_sources)

    def forward(self, x):
        # ===== Feature extraction =====
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))

        # ===== NOTES MODE =====
        if self.mode == "notes":
            x = self.final_conv(x)        # (B, num_notes, F, T)
            x = x.mean(dim=[2, 3])        # Global Avg Pool → (B, num_notes)
            return x

        # ===== INSTRUMENT MODE =====
        elif self.mode == "instrument":
            # Global average pooling
            x = x.mean(dim=[2, 3])        # (B, 256)

            family_logits = self.family_head(x)   # (B, num_families)
            source_logits = self.source_head(x)   # (B, 3)

            return {
                "family": family_logits,
                "source": source_logits
            }

    

if __name__ == "__main__":
    model = SimpleCNN()
    out = model(x)  # (1, 88, freq_reduced, time_reduced)
    piano_roll = out.max(dim=2).values  # max over freq axis
    plt.imshow(piano_roll[0].detach().numpy(), aspect='auto', cmap='gray')
    plt.xlabel("Frames temporelles")
    plt.ylabel("Notes (0-87)")
    plt.title("Piano-roll estimé")
    plt.show()

    