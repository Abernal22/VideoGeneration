import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import UCF101Dataset
from model import VideoCNN


def set_seed(seed: int = 42):
    """
    Make the experiment as reproducible as possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """
    Read training settings from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Train a 3D CNN on UCF101."
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Maximum number of videos to use.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    set_seed(args.seed)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    full_dataset = UCF101Dataset(
        video_dir="data/UCF-101",
        split_file="data/splits/trainlist01.txt",
        num_frames=16,
        image_size=112,
    )

    print(
        "Total videos available:",
        len(full_dataset),
    )

    # --------------------------------------------------
    # Optional dataset limit
    # --------------------------------------------------

    if args.samples is not None:

        if args.samples < 2:
            raise ValueError(
                "--samples must be at least 2."
            )

        if args.samples > len(full_dataset):
            print(
                f"Requested {args.samples} samples, "
                f"but only {len(full_dataset)} are available."
            )

            args.samples = len(full_dataset)

        full_dataset, _ = random_split(
            full_dataset,
            [
                args.samples,
                len(full_dataset) - args.samples,
            ],
            generator=torch.Generator().manual_seed(
                args.seed
            ),
        )

        print(
            "Videos used:",
            len(full_dataset),
        )

    # --------------------------------------------------
    # Train / validation split
    # --------------------------------------------------

    validation_size = max(
        1,
        int(0.10 * len(full_dataset)),
    )

    training_size = (
        len(full_dataset) - validation_size
    )

    train_dataset, validation_dataset = random_split(
        full_dataset,
        [
            training_size,
            validation_size,
        ],
        generator=torch.Generator().manual_seed(
            args.seed
        ),
    )

    print(
        "Training videos:",
        len(train_dataset),
    )

    print(
        "Validation videos:",
        len(validation_dataset),
    )

    # --------------------------------------------------
    # Data loaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = VideoCNN(
        num_classes=101
    )

    model = model.to(device)

    # --------------------------------------------------
    # Loss and optimizer
    # --------------------------------------------------

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # --------------------------------------------------
    # Training settings
    # --------------------------------------------------

    best_validation_accuracy = 0.0

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(args.epochs):

        # ==============================================
        # Training
        # ==============================================

        model.train()

        training_loss = 0.0
        training_correct = 0
        training_total = 0

        for batch_index, (
            videos,
            labels,
            _,
        ) in enumerate(train_loader):

            # Dataset format:
            # [B, T, C, H, W]
            #
            # Model format:
            # [B, C, T, H, W]

            videos = videos.permute(
                0,
                2,
                1,
                3,
                4,
            )

            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            predictions = model(videos)

            loss = loss_function(
                predictions,
                labels,
            )

            loss.backward()

            optimizer.step()

            training_loss += (
                loss.item()
                * labels.size(0)
            )

            predicted_classes = (
                predictions.argmax(dim=1)
            )

            training_correct += (
                predicted_classes == labels
            ).sum().item()

            training_total += (
                labels.size(0)
            )

            if (batch_index + 1) % 50 == 0:
                print(
                    f"  Batch "
                    f"[{batch_index + 1}/"
                    f"{len(train_loader)}]"
                )

        training_loss /= training_total

        training_accuracy = (
            training_correct
            / training_total
        )

        # ==============================================
        # Validation
        # ==============================================

        model.eval()

        validation_loss = 0.0
        validation_correct = 0
        validation_total = 0

        with torch.no_grad():

            for videos, labels, _ in validation_loader:

                videos = videos.permute(
                    0,
                    2,
                    1,
                    3,
                    4,
                )

                videos = videos.to(device)
                labels = labels.to(device)

                predictions = model(videos)

                loss = loss_function(
                    predictions,
                    labels,
                )

                validation_loss += (
                    loss.item()
                    * labels.size(0)
                )

                predicted_classes = (
                    predictions.argmax(dim=1)
                )

                validation_correct += (
                    predicted_classes == labels
                ).sum().item()

                validation_total += (
                    labels.size(0)
                )

        validation_loss /= validation_total

        validation_accuracy = (
            validation_correct
            / validation_total
        )

        # ==============================================
        # Results
        # ==============================================

        print()
        print(
            f"Epoch [{epoch + 1:02d}/"
            f"{args.epochs}]"
        )

        print(
            f"  Train Loss:     "
            f"{training_loss:.4f}"
        )

        print(
            f"  Train Accuracy: "
            f"{training_accuracy:.4f}"
        )

        print(
            f"  Val Loss:       "
            f"{validation_loss:.4f}"
        )

        print(
            f"  Val Accuracy:   "
            f"{validation_accuracy:.4f}"
        )

        # ==============================================
        # Save best checkpoint
        # ==============================================

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = (
                validation_accuracy
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "validation_accuracy":
                        validation_accuracy,
                    "samples":
                        len(full_dataset),
                    "batch_size":
                        args.batch_size,
                    "learning_rate":
                        args.learning_rate,
                    "seed":
                        args.seed,
                },
                "best_ucf101_model.pth",
            )

            print(
                "  Saved new best model."
            )

        print()


if __name__ == "__main__":
    main()