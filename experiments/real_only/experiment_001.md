# Experiment 001 — UCF101 Real-Video Smoke Test

## Objective

Verify that the UCF101 video classification pipeline can successfully
load videos, sample temporal frames, train a 3D CNN, and perform
validation on the CPU.

## Dataset

UCF101

Training split:
`data/splits/trainlist01.txt`

Number of videos available:
9537

Smoke-test subset:
200 videos

Training subset:
180 videos

Validation subset:
20 videos

## Preprocessing

Frames per video:
16

Image resolution:
112 x 112

Channels:
RGB

Video tensor format:
[T, C, H, W]

Model input format:
[B, C, T, H, W]

## Model

Model:
Custom 3D CNN

Number of output classes:
101

## Training

Device:
CPU

Batch size:
2

Epochs:
1

Optimizer:
Adam

Learning rate:
0.001

Loss:
CrossEntropyLoss

Random seed:
42

## Results

Training Loss:
4.6418

Training Accuracy:
0.0111

Validation Loss:
4.6702

Validation Accuracy:
0.0000

## Interpretation

This experiment was intended as an engineering smoke test rather than
a final performance evaluation.

The model successfully completed the complete training and validation
pipeline without video decoding or tensor-shape errors.

The low accuracy is expected because the experiment used only 200 videos
and one training epoch across 101 action classes.

## Next Step

Establish a larger real-video baseline using the official UCF101
training and test splits.# Experiment 001 — UCF101 Real-Video Smoke Test

## Objective

Verify that the UCF101 video classification pipeline can successfully
load videos, sample temporal frames, train a 3D CNN, and perform
validation on the CPU.

## Dataset

UCF101

Training split:
`data/splits/trainlist01.txt`

Number of videos available:
9537

Smoke-test subset:
200 videos

Training subset:
180 videos

Validation subset:
20 videos

## Preprocessing

Frames per video:
16

Image resolution:
112 x 112

Channels:
RGB

Video tensor format:
[T, C, H, W]

Model input format:
[B, C, T, H, W]

## Model

Model:
Custom 3D CNN

Number of output classes:
101

## Training

Device:
CPU

Batch size:
2

Epochs:
1

Optimizer:
Adam

Learning rate:
0.001

Loss:
CrossEntropyLoss

Random seed:
42

## Results

Training Loss:
4.6418

Training Accuracy:
0.0111

Validation Loss:
4.6702

Validation Accuracy:
0.0000

## Interpretation

This experiment was intended as an engineering smoke test rather than
a final performance evaluation.

The model successfully completed the complete training and validation
pipeline without video decoding or tensor-shape errors.

The low accuracy is expected because the experiment used only 200 videos
and one training epoch across 101 action classes.

## Next Step

Establish a larger real-video baseline using the official UCF101
training and test splits.
