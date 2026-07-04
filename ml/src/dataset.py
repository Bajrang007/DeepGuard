"""
PyTorch Dataset for loading sequences of face-crop frames extracted by
extract_faces.py. Groups frames from each video into fixed-length
sequences (e.g., 16 frames) for the EfficientNet+GRU temporal model.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

SEQUENCE_LENGTH = 16
IMAGE_SIZE = 224


class DeepfakeSequenceDataset(Dataset):
    def __init__(self, root_dir, label_map, sequence_length=SEQUENCE_LENGTH):
        """
        root_dir: folder containing label subfolders, e.g.
            root_dir/real/000/frame_0000.jpg, frame_0001.jpg, ...
            root_dir/fake_deepfakes/000_003/frame_0000.jpg, ...
        label_map: dict mapping folder name -> integer label, e.g.
            {"real": 0, "fake_deepfakes": 1, "fake_face2face": 1, ...}
        """
        self.sequence_length = sequence_length
        self.samples = []  # list of (list_of_frame_paths, label)

        for folder_name, label in label_map.items():
            category_dir = os.path.join(root_dir, folder_name)
            if not os.path.isdir(category_dir):
                continue

            for video_id in os.listdir(category_dir):
                video_dir = os.path.join(category_dir, video_id)
                frame_files = sorted(
                    f for f in os.listdir(video_dir) if f.endswith(".jpg")
                )
                frame_paths = [os.path.join(video_dir, f) for f in frame_files]

                # Split into non-overlapping windows of `sequence_length`
                for i in range(0, len(frame_paths) - sequence_length + 1, sequence_length):
                    window = frame_paths[i:i + sequence_length]
                    self.samples.append((window, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        frames = []
        for path in frame_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img.astype(np.float32) / 255.0
            frames.append(img)

        # Shape: (sequence_length, H, W, C) -> (sequence_length, C, H, W)
        frames = np.stack(frames)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)

        return frames, torch.tensor(label, dtype=torch.long)