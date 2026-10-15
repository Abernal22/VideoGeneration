import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.models.video import r3d_18, R3D_18_Weights

from dataset import UCF101Dataset


def prepare_video(videos, device):
    # Dataset: [B, T, C, H, W]
    # R3D-18: [B, C, T, H, W]
    videos = videos.to(device).permute(0, 2, 1, 3, 4)

    mean = torch.tensor(
        [0.43216, 0.394666, 0.37645], device=device
    ).view(1, 3, 1, 1, 1)

    std = torch.tensor(
        [0.22803, 0.22145, 0.216989], device=device
    ).view(1, 3, 1, 1, 1)

    return (videos - mean) / std


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None

    # Keep the pretrained backbone's batch-normalization layers fixed.
    model.eval()
    model.fc.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    for videos, labels, _ in loader:
        labels = labels.to(device)
        videos = prepare_video(videos, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(videos)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    if args.samples < 2 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Use samples >= 2, epochs >= 1, batch-size >= 1.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Seed: {args.seed}")

    dataset = UCF101Dataset(
        video_dir="data/UCF-101",
        split_file="data/splits/trainlist01.txt",
        num_frames=16,
        image_size=112,
    )

    count = min(args.samples, len(dataset))
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    subset = Subset(dataset, indices)

    train_size = max(1, int(0.9 * count))
    val_size = count - train_size
    if val_size == 0:
        train_size -= 1
        val_size = 1

    train_set, val_set = random_split(
        subset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(f"Videos selected: {count}")
    print(f"Training videos: {train_size}")
    print(f"Validation videos: {val_size}")
    print("Loading pretrained R3D-18...")

    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, 101)

    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.fc.parameters(), lr=args.learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path("results/checkpoints")
    metrics_dir = Path("results/real_only")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "r3d18_best.pth"
    metrics_path = metrics_dir / "r3d18_metrics.json"

    history = []
    best_accuracy = -1.0

    print(f"Trainable parameters: {sum(p.numel() for p in model.fc.parameters())}")
    print("Starting training...")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device
        )

        elapsed = time.time() - start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "seconds": round(elapsed, 2),
        }
        history.append(record)

        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.1f} seconds)")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Train accuracy: {train_accuracy:.4f}")
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  Val accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_accuracy,
                    "weights": str(weights),
                    "seed": args.seed,
                    "samples": count,
                    "num_frames": 16,
                    "image_size": 112,
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint: {checkpoint_path}")

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "architecture": "R3D-18",
                    "pretrained_weights": str(weights),
                    "dataset": "UCF101",
                    "split": "random development split from trainlist01.txt",
                    "device": str(device),
                    "config": vars(args),
                    "history": history,
                },
                file,
                indent=2,
            )

    print("\nExperiment complete.")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print("Note: this is a development run, not a final test result.")


if __name__ == "__main__":
    main()
