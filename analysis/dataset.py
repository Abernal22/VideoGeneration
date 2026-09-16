from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class UCF101Dataset(Dataset):
    """
    UCF101 video dataset.

    Each sample returns:

        video:
            Tensor with shape [T, C, H, W]

        label:
            Integer from 0 to 100

        class_name:
            Name of the action class.
    """

    def __init__(
        self,
        video_dir: str = "data/UCF-101",
        split_file: str = "data/splits/trainlist01.txt",
        num_frames: int = 16,
        image_size: int = 112,
    ):
        self.video_dir = Path(video_dir)
        self.split_file = Path(split_file)
        self.num_frames = num_frames
        self.image_size = image_size

        # --------------------------------------------------
        # Load class names
        # --------------------------------------------------

        class_file = self.split_file.parent / "classInd.txt"

        if not class_file.exists():
            raise FileNotFoundError(
                f"Could not find classInd.txt at {class_file}"
            )

        self.class_to_index = {}

        with open(class_file, "r") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                index, class_name = line.split(maxsplit=1)

                self.class_to_index[class_name] = int(index) - 1

        # --------------------------------------------------
        # Load video paths
        # --------------------------------------------------

        if not self.split_file.exists():
            raise FileNotFoundError(
                f"Could not find split file at {self.split_file}"
            )

        self.samples = []

        with open(self.split_file, "r") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                relative_path = parts[0]

                class_name = Path(relative_path).parts[0]

                video_path = self.video_dir / relative_path

                if not video_path.exists():
                    continue

                label = self.class_to_index[class_name]

                self.samples.append(
                    (
                        video_path,
                        label,
                        class_name,
                    )
                )

        if not self.samples:
            raise RuntimeError(
                "No valid UCF101 videos were found."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, label, class_name = self.samples[index]

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        # --------------------------------------------------
        # Determine number of frames
        # --------------------------------------------------

        reported_frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # Some AVI files report one more frame than can
        # actually be decoded. We therefore do not trust
        # the reported value completely.
        if reported_frame_count <= 0:
            reported_frame_count = self.num_frames

        # Create approximate sampling positions.
        sample_positions = torch.linspace(
            0,
            max(reported_frame_count - 1, 0),
            self.num_frames,
        ).long()

        sample_positions = set(
            int(position)
            for position in sample_positions
        )

        # --------------------------------------------------
        # Sequential decoding
        # --------------------------------------------------

        selected_frames = []

        frame_index = 0

        while True:
            success, frame = capture.read()

            if not success:
                break

            if frame_index in sample_positions:

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                frame = cv2.resize(
                    frame,
                    (
                        self.image_size,
                        self.image_size,
                    ),
                    interpolation=cv2.INTER_AREA,
                )

                frame = torch.from_numpy(
                    frame
                )

                frame = frame.permute(
                    2,
                    0,
                    1,
                )

                frame = frame.float() / 255.0

                selected_frames.append(frame)

            frame_index += 1

        capture.release()

        # --------------------------------------------------
        # Handle videos where fewer frames were decoded
        # --------------------------------------------------

        if not selected_frames:
            raise RuntimeError(
                f"Could not read frames from {video_path}"
            )

        while len(selected_frames) < self.num_frames:
            selected_frames.append(
                selected_frames[-1].clone()
            )

        # If the video produced slightly more frames than
        # expected, keep exactly num_frames.
        selected_frames = selected_frames[
            :self.num_frames
        ]

        # --------------------------------------------------
        # Stack frames
        # --------------------------------------------------

        video = torch.stack(
            selected_frames
        )

        return (
            video,
            label,
            class_name,
        )