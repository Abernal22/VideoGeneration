import torch

from dataset import VideoDataset
from model import VideoCNN


def main():
    dataset = VideoDataset(
        video_dir="data",
        num_frames=16,
        image_size=112,
    )

    video, filename = dataset[0]

    print("Input video:", filename)
    print("Original shape:", video.shape)

    # Add batch dimension.
    video = video.unsqueeze(0)

    # Convert:
    #
    # [batch, frames, channels, height, width]
    #
    # to:
    #
    # [batch, channels, frames, height, width]

    video = video.permute(0, 2, 1, 3, 4)

    print("Model input shape:", video.shape)

    model = VideoCNN()

    prediction = model(video)

    print("Prediction shape:", prediction.shape)
    print("Prediction:", prediction)


if __name__ == "__main__":
    main()