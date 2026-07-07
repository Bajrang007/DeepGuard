"""
PyTorch Dataset for ASVspoof 2019 LA audio anti-spoofing data.
Uses soundfile for loading (avoids torchcodec/FFmpeg dependency issues).
"""

import os
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = SAMPLE_RATE * 4  # 4 seconds, padded/truncated to fixed length


def parse_protocol_file(protocol_path):
    """
    Parses an ASVspoof protocol file into a list of (filename, label) tuples.
    label: 0 = bonafide (real), 1 = spoof (fake)
    """
    entries = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            filename = parts[1]
            label_str = parts[4]
            label = 0 if label_str == "bonafide" else 1
            entries.append((filename, label))
    return entries


class ASVspoofDataset(Dataset):
    def __init__(self, protocol_path, audio_dir, sample_rate=SAMPLE_RATE, max_length=MAX_AUDIO_LENGTH):
        self.entries = parse_protocol_file(protocol_path)
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.max_length = max_length

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        filename, label = self.entries[idx]
        audio_path = os.path.join(self.audio_dir, f"{filename}.flac")

        data, sr = sf.read(audio_path, dtype="float32")

        # Convert to mono if stereo
        if data.ndim > 1:
            data = data.mean(axis=1)

        waveform = torch.from_numpy(data)

        # Resample if needed (simple linear interpolation - ASVspoof is already 16kHz, so
        # this branch normally won't trigger, but included for safety)
        if sr != self.sample_rate:
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0).unsqueeze(0),
                scale_factor=self.sample_rate / sr,
                mode="linear",
                align_corners=False,
            ).squeeze()

        # Pad or truncate to fixed length
        if waveform.shape[0] > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            pad_amount = self.max_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    protocol = "ml/data/raw/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    audio_dir = "ml/data/raw/asvspoof2019/LA/ASVspoof2019_LA_train/flac"

    dataset = ASVspoofDataset(protocol, audio_dir)
    print(f"Total samples: {len(dataset)}")

    waveform, label = dataset[0]
    print(f"Waveform shape: {waveform.shape}")
    print(f"Label: {label.item()} (0=bonafide/real, 1=spoof/fake)")