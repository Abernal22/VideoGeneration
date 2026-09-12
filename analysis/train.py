import torch
import torch.nn as nn
import torch.optim as optim

from dataset import VideoDataset
from model import VideoCNN


def main():
    # ---------------------------------------------------------
    # 1. Load the video dataset
    # ---------------------------------------------------------

    dataset = VideoDataset(
        video_dir="data",
        num_frames=16,
        image_size=112,
    )

    video, filename = dataset[0]

    print("Training video:", filename)

    # Add batch dimension.
    #
    # [frames, channels, height, width]
    #          ↓
    # [batch, frames, channels, height, width]

    video = video.unsqueeze(0)

    # Conv3D expects:
    #
    # [batch, channels, frames, height, width]

    video = video.permute(0, 2, 1, 3, 4)

    print("Input shape:", video.shape)

    # ---------------------------------------------------------
    # 2. Create a temporary target
    # ---------------------------------------------------------

    # This is NOT a real medical label.
    #
    # We are only testing the training pipeline.

    target = torch.tensor([[0.5]], dtype=torch.float32)

    # ---------------------------------------------------------
    # 3. Create the model
    # ---------------------------------------------------------

    model = VideoCNN()

    # ---------------------------------------------------------
    # 4. Define loss function
    # ---------------------------------------------------------

    loss_function = nn.MSELoss()

    # ---------------------------------------------------------
    # 5. Define optimizer
    # ---------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # ---------------------------------------------------------
    # 6. Training loop
    # ---------------------------------------------------------

    epochs = 20

    for epoch in range(epochs):

        # Put model into training mode.
        model.train()

        # Clear old gradients.
        optimizer.zero_grad()

        # Forward pass.
        prediction = model(video)

        # Calculate loss.
        loss = loss_function(
            prediction,
            target,
        )

        # Calculate gradients.
        loss.backward()

        # Update model weights.
        optimizer.step()

        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] "
            f"Loss: {loss.item():.6f} "
            f"Prediction: {prediction.item():.4f}"
        )


if __name__ == "__main__":
    main()