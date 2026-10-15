import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.models.video import r3d_18, R3D_18_Weights

from dataset import UCF101Dataset


def prepare_video(videos, device):
    # Dataset output: [B, T, C, H, W]
    # R3D-18 input:  [B, C, T, H, W]
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
    model.eval()
    model.fc.train(training)

    loss_sum = 0.0
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

        size = labels.size(0)
        loss_sum += loss.item() * size
        correct += (outputs.argmax(1) == labels).sum().item()
        total += size

    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    if args.samples_per_class < 2:
        raise ValueError("samples-per-class must be at least 2.")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    dataset = UCF101Dataset(
        video_dir="data/UCF-101",
        split_file="data/splits/trainlist01.txt",
        num_frames=16,
        image_size=112,
    )

    # Group videos by class and sample the same number per class.
    class_indices = defaultdict(list)
    for index, (_, _, class_name) in enumerate(dataset.samples):
        class_indices[class_name].append(index)

    rng = random.Random(args.seed)
    train_indices = []
    val_indices = []

    for class_name in sorted(class_indices):
        candidates = class_indices[class_name].copy()
        rng.shuffle(candidates)

        if len(candidates) < args.samples_per_class:
            raise ValueError(
                f"Not enough videos for class {class_name}"
            )

        selected = candidates[:args.samples_per_class]
        split_at = max(1, int(0.8 * len(selected)))
        split_at = min(split_at, len(selected) - 1)

        train_indices.extend(selected[:split_at])
        val_indices.extend(selected[split_at:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_counts = Counter(
        dataset.samples[i][2] for i in train_indices
    )
    val_counts = Counter(
        dataset.samples[i][2] for i in val_indices
    )

    print(f"Training videos: {len(train_set)}")
    print(f"Validation videos: {len(val_set)}")
    print(f"Training classes: {len(train_counts)} / 101")
    print(f"Validation classes: {len(val_counts)} / 101")
    print(f"Train examples per class: {sorted(set(train_counts.values()))}")
    print(f"Validation examples per class: {sorted(set(val_counts.values()))}")
    print("Official test split will not be used in this experiment.")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )

    weights = R3D_18_Weights.DEFAULT
    print("Loading pretrained R3D-18...")
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

    checkpoint_path = checkpoint_dir / "r3d18_stratified_best.pth"
    metrics_path = metrics_dir / "r3d18_stratified_metrics.json"

    history = []
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "seconds": round(time.time() - start, 2),
        }
        history.append(record)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  Val accuracy: {val_acc:.4f}")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "weights": str(weights),
                    "seed": args.seed,
                    "samples_per_class": args.samples_per_class,
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
                    "training_split": "trainlist01.txt",
                    "validation_protocol": (
                        "stratified subset of trainlist01.txt; "
                        "official test set held out"
                    ),
                    "device": str(device),
                    "config": vars(args),
                    "training_class_counts": dict(train_counts),
                    "validation_class_counts": dict(val_counts),
                    "history": history,
                },
                file,
                indent=2,
            )

    print("\nExperiment complete.")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Metrics: {metrics_path}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
