from dataset import UCF101Dataset


def main():
    dataset = UCF101Dataset(
        video_dir="data/UCF-101",
        split_file="data/splits/trainlist01.txt",
        num_frames=16,
        image_size=112,
    )

    print("Number of videos:", len(dataset))

    video, label, class_name = dataset[0]

    print("Video shape:", video.shape)
    print("Label:", label)
    print("Class:", class_name)
    print("Minimum:", video.min().item())
    print("Maximum:", video.max().item())


if __name__ == "__main__":
    main()
