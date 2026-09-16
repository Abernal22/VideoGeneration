import torch

from dataset import UCF101Dataset
from model import VideoCNN


def main():
    dataset = UCF101Dataset(
        video_dir="data/UCF-101",
        split_file="data/splits/trainlist01.txt",
        num_frames=16,
        image_size=112,
    )

    video, label, class_name = dataset[0]

    print("Video class:", class_name)
    print("Ground-truth label:", label)
    print("Original shape:", video.shape)

    # Add batch dimension.
    video = video.unsqueeze(0)

    # [batch, frames, channels, height, width]
    # ->
    # [batch, channels, frames, height, width]
    video = video.permute(0, 2, 1, 3, 4)

    print("Model input shape:", video.shape)

    model = VideoCNN(num_classes=101)

    prediction = model(video)

    print("Prediction shape:", prediction.shape)

    # Find the class with the largest score.
    predicted_class = prediction.argmax(dim=1).item()

    print("Predicted class index:", predicted_class)


if __name__ == "__main__":
    main()