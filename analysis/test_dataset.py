from dataset import VideoDataset


def main():
    dataset = VideoDataset(
        video_dir="data",
        num_frames=16,
        image_size=112,
    )

    print("Number of videos:", len(dataset))

    video, filename = dataset[0]

    print("Filename:", filename)
    print("Tensor shape:", video.shape)
    print("Minimum:", video.min().item())
    print("Maximum:", video.max().item())


if __name__ == "__main__":
    main()