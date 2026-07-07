"""
Efficient training script: mixed precision (AMP), video-level train/val
split, class-weighted loss, per-epoch checkpointing to Drive.
"""

import os
import sys
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
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


def compute_class_weights(dataset, num_classes=2):
    counts = [0] * num_classes
    for sample in dataset.samples:
        counts[sample[1]] += 1
    total = sum(counts)
    return torch.tensor([total / c if c > 0 else 0.0 for c in counts], dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for frames, labels in tqdm(loader, desc="Training"):
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(device_type="cuda"):
            outputs = model(frames)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(loader),
            accuracy_score(all_labels, all_preds),
            roc_auc_score(all_labels, all_preds))


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Validation"):
            frames, labels = frames.to(device), labels.to(device)
            with autocast(device_type="cuda"):
                outputs = model(frames)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(loader),
            accuracy_score(all_labels, all_preds),
            roc_auc_score(all_labels, all_preds))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-dir", default="ml/checkpoints")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    wandb.init(project="deepguard", name="phase2-video-training-v2", job_type="train")
    wandb.config.update(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_map = get_label_map()
    train_videos, val_videos = split_videos(args.data_dir, label_map)
    print(f"Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")

    train_dataset = DeepfakeSequenceDataset(args.data_dir, label_map, 16, allowed_videos=train_videos)
    val_dataset = DeepfakeSequenceDataset(args.data_dir, label_map, 16, allowed_videos=val_videos)
    print(f"Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = DeepfakeVideoModel().to(device)
    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"Class weights (real, fake): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    start_epoch, best_val_auc = 0, 0.0

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_auc = ckpt.get("best_val_auc", 0.0)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc, train_auc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_auc = eval_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        wandb.log({
            "train_loss": train_loss, "train_accuracy": train_acc, "train_auc": train_auc,
            "val_loss": val_loss, "val_accuracy": val_acc, "val_auc": val_auc,
        })

        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "best_val_auc": best_val_auc,
        }, ckpt_path)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))
            print(f"Saved best model (AUC: {best_val_auc:.4f})")

    wandb.finish()
    print(f"\nTraining complete. Best validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
