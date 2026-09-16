import torch
import torch.nn as nn


class VideoCNN(nn.Module):
    """
    Simple 3D CNN for UCF101 video classification.

    Input:
        [batch, channels, frames, height, width]

    Example:
        [8, 3, 16, 112, 112]

    Output:
        [batch, 101]

    Each output corresponds to one UCF101 action class.
    """

    def __init__(self, num_classes: int = 101):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.MaxPool3d(
                kernel_size=2,
            ),

            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.MaxPool3d(
                kernel_size=2,
            ),

            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d(
                output_size=(1, 1, 1),
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                in_features=64,
                out_features=32,
            ),

            nn.ReLU(),

            nn.Linear(
                in_features=32,
                out_features=num_classes,
            ),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x