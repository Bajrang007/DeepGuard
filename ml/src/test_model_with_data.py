"""
Tests the DeepfakeVideoModel using a real sample from DeepfakeSequenceDataset,
confirming the full pipeline (data loading -> model forward pass) works.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
from dataset import DeepfakeSequenceDataset
from model import DeepfakeVideoModel

TEST_DIR = "ml/data/processed/faceforensics_faces"
LABEL_MAP = {
    "real": 0,
    "fake_deepfakes": 1,
    "fake_face2face": 1,
    "fake_faceshifter": 1,
    "fake_faceswap": 1,
    "fake_neuraltextures": 1,
}

if __name__ == "__main__":
    dataset = DeepfakeSequenceDataset(TEST_DIR, LABEL_MAP, sequence_length=16)
    print(f"Dataset size: {len(dataset)}")

    frames, label = dataset[0]
    frames = frames.unsqueeze(0)  # add batch dimension: (1, 16, 3, 224, 224)

    model = DeepfakeVideoModel()
    model.eval()

    with torch.no_grad():
        output = model(frames)

    print("Model output (raw logits):", output)
    print("Predicted class:", torch.argmax(output, dim=1).item())
    print("Actual label:", label.item())