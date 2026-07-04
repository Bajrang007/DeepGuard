"""
Training script for DeepGuard video deepfake detection model.
Designed to run on Colab with GPU + persistent checkpointing to Google Drive.

Usage (in Colab):
1. Mount Drive: from google.colab import drive; drive.mount('/content/drive')
2. Clone repo or upload ml/ folder
3. Run: python train.py --data-dir /path/to/extracted/faces --checkpoint-dir /content/drive/MyDrive/deepguard_checkpoints
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

sys.path.append(os.path.dirname(__file__))
from dataset import DeepfakeSequenceDataset
from model import DeepfakeVideoModel


def get_label_map():
    """All 6 categories, with all non-real as fake (label=1)"""
    return {
        "real": 0,
        "fake_deepfakes": 1,
        "fake_face2face": 1,
        "fake_faceshifter": 1,
        "fake_faceswap": 1,
        "fake_neuraltextures": 1,
    }


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for frames, labels in tqdm(train_loader, desc="Training"):
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, accuracy, auc


def eval_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in tqdm(val_loader, desc="Validation"):
            frames, labels = frames.to(device), labels.to(device)

            outputs = model(frames)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, accuracy, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path to extracted faces directory")
    parser.add_argument("--checkpoint-dir", default="ml/checkpoints", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--resume-from", default=None, help="Resume from checkpoint (path)")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize W&B
    wandb.init(project="deepguard", name="phase2-video-training", job_type="train")
    wandb.config.update(args)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset and DataLoader
    print("Loading dataset...")
    dataset = DeepfakeSequenceDataset(args.data_dir, get_label_map(), sequence_length=16)
    print(f"Total sequences: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = DeepfakeVideoModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val_auc = 0.0

    # Resume from checkpoint if provided
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_auc = checkpoint.get("best_val_auc", 0.0)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = eval_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_auc": train_auc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_auc": val_auc,
        })

        # Save checkpoint every epoch
        checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_auc": best_val_auc,
        }, checkpoint_path)

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model (AUC: {best_val_auc:.4f})")

    wandb.finish()
    print(f"\nTraining complete. Best validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()