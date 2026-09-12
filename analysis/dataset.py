from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    Dataset for loading videos and sampling a fixed number of frames.

    Each video is returned as a tensor with shape:

        [T, C, H, W]

    T = number of frames
    C = RGB channels
    H = height
    W = width
    """

    def __init__(
        self,
        video_dir: str,
        num_frames: int = 16,
        image_size: int = 112,
    ):
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.image_size = image_size

        self.video_files = sorted(
            list(self.video_dir.glob("*.mp4"))
            + list(self.video_dir.glob("*.avi"))
            + list(self.video_dir.glob("*.mov"))
        )

        if not self.video_files:
            raise RuntimeError(
                f"No video files found in {self.video_dir}"
            )

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        video_path = self.video_files[index]

        capture = cv2.VideoCapture(str(video_path))

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        total_frames = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            capture.release()

            raise RuntimeError(
                f"Video contains no frames: {video_path}"
            )

        # Select frames evenly throughout the video.
        frame_indices = torch.linspace(
            0,
            total_frames - 1,
            self.num_frames,
        ).long()

        frames = []

        for frame_index in frame_indices:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index),
            )

            success, frame = capture.read()

            if not success:
                capture.release()

                raise RuntimeError(
                    f"Could not read frame {frame_index}"
                )

            # OpenCV loads images as BGR.
            # Convert BGR -> RGB.
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            # Resize the frame.
            frame = cv2.resize(
                frame,
                (self.image_size, self.image_size),
            )

            # Convert NumPy array -> PyTorch tensor.
            frame = torch.from_numpy(frame)

            # H x W x C -> C x H x W
            frame = frame.permute(2, 0, 1)

            # Convert 0-255 -> 0.0-1.0
            frame = frame.float() / 255.0

            frames.append(frame)

        capture.release()

        # T x C x H x W
        video = torch.stack(frames)

        return video, video_path.name