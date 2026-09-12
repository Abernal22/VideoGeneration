import torch
import torch.nn as nn


class VideoCNN(nn.Module):
    """
    A simple 3D convolutional neural network for video analysis.

    Input:
        [batch, channels, frames, height, width]

    Example:
        [1, 3, 16, 112, 112]

    Output:
        [batch, 1]
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # First convolution
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

            # Second convolution
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

            # Third convolution
            nn.Conv3d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            # Reduce the feature map to one value per channel
            nn.AdaptiveAvgPool3d(
                output_size=(1, 1, 1),
            ),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                in_features=64,
                out_features=32,
            ),

            nn.ReLU(),

            nn.Linear(
                in_features=32,
                out_features=1,
            ),
        )

    def forward(self, x):
        """
        Run a video through the network.
        """

        x = self.features(x)
        x = self.regressor(x)

        return x