"""
PyTorch Dataset for loading sequences of face-crop frames extracted by
extract_faces.py. Groups frames from each video into fixed-length
sequences (e.g., 16 frames) for the EfficientNet+GRU temporal model.

Supports splitting by VIDEO (not by sequence) to avoid data leakage
between train and validation sets.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

SEQUENCE_LENGTH = 16
IMAGE_SIZE = 224


def get_all_video_keys(root_dir, label_map):
    """
    Returns a sorted list of unique video keys, e.g. "real/000",
    "fake_deepfakes/003". Used to split videos into train/val BEFORE
    building sequences, so no video's frames end up in both sets.
    """
    video_keys = []
    for folder_name in label_map:
        category_dir = os.path.join(root_dir, folder_name)
        if not os.path.isdir(category_dir):
            continue
        for video_id in os.listdir(category_dir):
            video_keys.append(f"{folder_name}/{video_id}")
    return sorted(video_keys)


class DeepfakeSequenceDataset(Dataset):
    def __init__(self, root_dir, label_map, sequence_length=SEQUENCE_LENGTH, allowed_videos=None):
        """
        root_dir: folder containing label subfolders, e.g.
            root_dir/real/000/frame_0000.jpg, frame_0001.jpg, ...
            root_dir/fake_deepfakes/000_003/frame_0000.jpg, ...
        label_map: dict mapping folder name -> integer label, e.g.
            {"real": 0, "fake_deepfakes": 1, "fake_face2face": 1, ...}
        allowed_videos: optional set of "folder_name/video_id" strings. If
            given, only videos in this set are included. Used to enforce a
            video-level train/val split (see get_all_video_keys above).
        """
        self.sequence_length = sequence_length
        self.samples = []  # list of (list_of_frame_paths, label, video_key, folder_name)

        for folder_name, label in label_map.items():
            category_dir = os.path.join(root_dir, folder_name)
            if not os.path.isdir(category_dir):
                continue

            for video_id in os.listdir(category_dir):
                video_key = f"{folder_name}/{video_id}"

                if allowed_videos is not None and video_key not in allowed_videos:
                    continue

                video_dir = os.path.join(category_dir, video_id)
                frame_files = sorted(
                    f for f in os.listdir(video_dir) if f.endswith(".jpg")
                )
                frame_paths = [os.path.join(video_dir, f) for f in frame_files]

                # Split into non-overlapping windows of `sequence_length`
                for i in range(0, len(frame_paths) - sequence_length + 1, sequence_length):
                    window = frame_paths[i:i + sequence_length]
                    self.samples.append((window, label, video_key, folder_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label, video_key, folder_name = self.samples[idx]

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
