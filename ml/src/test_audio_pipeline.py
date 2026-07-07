"""
Tests RawNet2 with a real sample from ASVspoofDataset, confirming the
full pipeline (real audio loading -> model forward pass) works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
from audio_dataset import ASVspoofDataset
from model_audio import RawNet2

PROTOCOL = "ml/data/raw/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
AUDIO_DIR = "ml/data/raw/asvspoof2019/LA/ASVspoof2019_LA_train/flac"

if __name__ == "__main__":
    dataset = ASVspoofDataset(PROTOCOL, AUDIO_DIR)
    print(f"Dataset size: {len(dataset)}")

    waveform, label = dataset[0]
    waveform = waveform.unsqueeze(0)  # add batch dimension: (1, 64000)

    model = RawNet2()
    model.eval()

    with torch.no_grad():
        output = model(waveform)

    print("Model output (raw logits):", output)
    print("Predicted class:", torch.argmax(output, dim=1).item())
    print("Actual label:", label.item())