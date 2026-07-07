"""
Evaluates the trained model separately on each manipulation type
(Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures) plus
overall real vs fake performance, using the same video-level val split
used during training.
"""

import os
import sys
import random
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.append(os.path.dirname(__file__))
from dataset import DeepfakeSequenceDataset, get_all_video_keys
from model import DeepfakeVideoModel


def get_label_map():
    return {
        "real": 0,
        "fake_deepfakes": 1,
        "fake_face2face": 1,
        "fake_faceshifter": 1,
        "fake_faceswap": 1,
        "fake_neuraltextures": 1,
    }


def split_videos(data_dir, label_map, val_fraction=0.2, seed=42):
    all_videos = get_all_video_keys(data_dir, label_map)
    rng = random.Random(seed)
    rng.shuffle(all_videos)
    val_size = int(len(all_videos) * val_fraction)
    return set(all_videos[val_size:]), set(all_videos[:val_size])


def main():
    data_dir = "/content/faceforensics_faces"
    checkpoint_path = "/content/drive/MyDrive/deepguard_checkpoints/best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = get_label_map()

    _, val_videos = split_videos(data_dir, label_map)
    val_dataset = DeepfakeSequenceDataset(data_dir, label_map, sequence_length=16, allowed_videos=val_videos)
    print(f"Total validation sequences: {len(val_dataset)}")

    model = DeepfakeVideoModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    results_by_type = defaultdict(lambda: {"preds": [], "labels": []})

    with torch.no_grad():
        for idx in range(len(val_dataset)):
            frame_paths, label, video_key, folder_name = val_dataset.samples[idx]
            frames, label_tensor = val_dataset[idx]
            frames = frames.unsqueeze(0).to(device)

            output = model(frames)
            pred = torch.argmax(output, dim=1).item()

            results_by_type[folder_name]["preds"].append(pred)
            results_by_type[folder_name]["labels"].append(label)

            if idx % 500 == 0:
                print(f"Processed {idx}/{len(val_dataset)}")

    print("\n=== Per-Category Results ===")
    for folder_name, data in sorted(results_by_type.items()):
        labels = data["labels"]
        preds = data["preds"]
        acc = accuracy_score(labels, preds)

        if len(set(labels)) > 1:
            auc = roc_auc_score(labels, preds)
            print(f"{folder_name:25s} | N={len(labels):5d} | Acc={acc:.4f} | AUC={auc:.4f}")
        else:
            detection_rate = sum(preds) / len(preds) if labels[0] == 1 else (len(preds) - sum(preds)) / len(preds)
            print(f"{folder_name:25s} | N={len(labels):5d} | Acc={acc:.4f} | Detection Rate={detection_rate:.4f}")


if __name__ == "__main__":
    main()
