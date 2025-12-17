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
    def __init__(self,num_notes = 88):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3,3), padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(3,3), padding=1)
        self.final_conv = nn.Conv2d(256, num_notes, kernel_size=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        # Final conv → num_notes channels
        x = self.final_conv(x)
        x = torch.sigmoid(x)
        return x  # shape: (batch, num_notes, freq_reduced, time_reduced)
    
model = SimpleCNN()
out = model(x)  # (1, 88, freq_reduced, time_reduced)
piano_roll = out.max(dim=2).values  # max over freq axis
plt.imshow(piano_roll[0].detach().numpy(), aspect='auto', cmap='gray')
plt.xlabel("Frames temporelles")
plt.ylabel("Notes (0-87)")
plt.title("Piano-roll estimé")
plt.show()


    