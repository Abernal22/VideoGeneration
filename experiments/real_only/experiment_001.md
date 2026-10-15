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

## Experiment 002 — 1,000 Video Development Run

### Configuration

Dataset:
UCF101 trainlist01

Videos:
1000

Training videos:
900

Validation videos:
100

Frames per video:
16

Resolution:
112 x 112

Batch size:
2

Epochs:
3

Learning rate:
0.001

Optimizer:
Adam

Model:
Custom 3D CNN

Device:
CPU

### Results

| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|------:|-----------:|---------------:|---------:|-------------:|
| 1 | 4.6326 | 0.0144 | 4.5914 | 0.0100 |
| 2 | 4.6109 | 0.0178 | 4.6079 | 0.0100 |
| 3 | 4.5944 | 0.0167 | 4.6491 | 0.0000 |

### Observation

The training loss decreased across the three epochs, indicating that
the model was able to optimize successfully. However, classification
accuracy remained close to the approximately 0.99% chance level
associated with 101 classes.

The experiment therefore validates the training pipeline but does not
yet provide a strong video-classification baseline.

### Next Step

Evaluate a standard video-classification architecture and establish
an official UCF101 train/test evaluation protocol.





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
