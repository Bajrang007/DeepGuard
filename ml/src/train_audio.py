"""
Training script for RawNet2 audio anti-spoofing model on ASVspoof 2019 LA.
Uses the dataset's official train/dev protocol split (no custom splitting needed).
Designed to run on Colab with GPU + checkpointing to Google Drive.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import numpy as np

sys.path.append(os.path.dirname(__file__))
from audio_dataset import ASVspoofDataset
from model_audio import RawNet2


def compute_eer(labels, scores):
    """
    Computes Equal Error Rate (EER) - the standard anti-spoofing metric.
    scores: model's predicted probability of being "spoof" (class 1)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    return eer


def compute_class_weights(dataset, num_classes=2):
    counts = [0] * num_classes
    for _, label in dataset.entries:
        counts[label] += 1
    total = sum(counts)
    return torch.tensor([total / c if c > 0 else 0.0 for c in counts], dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, all_preds, all_labels, all_scores = 0.0, [], [], []

    for waveforms, labels in tqdm(loader, desc="Training"):
        waveforms, labels = waveforms.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast(device_type="cuda"):
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(probs)

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores)
    eer = compute_eer(all_labels, all_scores)

    return avg_loss, acc, auc, eer


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_scores = 0.0, [], [], []

    with torch.no_grad():
        for waveforms, labels in tqdm(loader, desc="Validation"):
            waveforms, labels = waveforms.to(device), labels.to(device)

            with autocast(device_type="cuda"):
                outputs = model(waveforms)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs)

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores)
    eer = compute_eer(all_labels, all_scores)

    return avg_loss, acc, auc, eer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-protocol", required=True)
    parser.add_argument("--train-audio-dir", required=True)
    parser.add_argument("--dev-protocol", required=True)
    parser.add_argument("--dev-audio-dir", required=True)
    parser.add_argument("--checkpoint-dir", default="ml/checkpoints_audio")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    wandb.init(project="deepguard", name="phase3-audio-training", job_type="train")
    wandb.config.update(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading train dataset...")
    train_dataset = ASVspoofDataset(args.train_protocol, args.train_audio_dir)
    print(f"Train samples: {len(train_dataset)}")

    print("Loading dev (validation) dataset...")
    val_dataset = ASVspoofDataset(args.dev_protocol, args.dev_audio_dir)
    print(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = RawNet2().to(device)

    class_weights = compute_class_weights(train_dataset).to(device)
    print(f"Class weights (bonafide, spoof): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    start_epoch, best_val_eer = 0, 1.0  # EER: lower is better, so start at worst possible (1.0)

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_eer = checkpoint.get("best_val_eer", 1.0)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc, train_auc, train_eer = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, val_auc, val_eer = eval_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}, EER: {train_eer:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}, EER: {val_eer:.4f}")

        wandb.log({
            "train_loss": train_loss, "train_accuracy": train_acc,
            "train_auc": train_auc, "train_eer": train_eer,
            "val_loss": val_loss, "val_accuracy": val_acc,
            "val_auc": val_auc, "val_eer": val_eer,
        })

        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "best_val_eer": best_val_eer,
        }, ckpt_path)

        # Lower EER is better
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model_audio.pth"))
            print(f"Saved best model (EER: {best_val_eer:.4f})")

    wandb.finish()
    print(f"\nTraining complete. Best validation EER: {best_val_eer:.4f}")


if __name__ == "__main__":
    main()