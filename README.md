# Video Generation and Analysis for Downstream Classification

## 1. Project Overview

This project investigates whether synthetically generated videos can improve downstream video analysis and action classification.

The main research question is:

**Can synthetic video data improve a video classification model's performance on real-world videos?**

The project is being developed in stages. The first stage establishes a real-video classification baseline using the UCF101 action recognition dataset. Later stages will investigate video generation, synthetic-data quality, and the effect of adding generated videos to the training set.

The goal is to evaluate not only the visual quality of generated videos, but also whether those videos preserve the temporal and semantic information needed for downstream machine learning tasks.

### Research objectives

- Establish a reproducible baseline for real-video classification.
- Evaluate pretrained video-recognition architectures using transfer learning.
- Investigate the effect of training-set size on classification performance.
- Develop a synthetic-video generation and evaluation pipeline.
- Compare real-only training against training with real and synthetic videos.
- Evaluate models on held-out real videos to measure generalization.

## 2. Current Project Status

| Component | Status |
|---|---|
| UCF101 dataset setup | Completed |
| Official dataset split files | Available locally |
| Video loading and preprocessing | Implemented |
| Basic 3D CNN | Implemented and tested |
| Pretrained R3D-18 classifier | Implemented |
| Stratified training experiments | Completed for initial configurations |
| Experiment metrics and checkpoints | Generated locally |
| Official test-set evaluation | Pending |
| Per-class metrics and confusion matrix | Pending |
| Synthetic-video generation | Planned |
| Real-only vs. real-plus-synthetic comparison | Planned |

The current focus is establishing a reliable real-only baseline before introducing synthetic data.

## 3. Repository Structure

The repository is organized into separate modules for video generation, video analysis, experiments, and evaluation.

```text
VideoGeneration/
├── analysis/
│   ├── dataset.py
│   ├── model.py
│   ├── test_model.py
│   ├── test_ucf101.py
│   ├── train.py
│   ├── train_r3d18.py
│   └── train_r3d18_stratified.py
│
├── configs/
│   ├── analysis/
│   ├── experiments/
│   └── generation/
│
├── data/
│   ├── splits/
│   │   ├── classInd.txt
│   │   ├── trainlist01.txt
│   │   ├── testlist01.txt
│   │   ├── trainlist02.txt
│   │   ├── testlist02.txt
│   │   ├── trainlist03.txt
│   │   └── testlist03.txt
│   ├── UCF-101/                 # Local dataset; not committed
│   └── test_video.mp4            # Local test video; not committed
│
├── experiments/
│   ├── real_only/
│   │   └── experiment_001.md
│   ├── real_plus_synthetic/
│   └── ablations/
│
├── generation/
│   ├── preprocess/
│   ├── generate/
│   └── evaluate_generation.py
│
├── evaluation/
│   ├── metrics/
│   ├── temporal/
│   ├── feature_similarity/
│   └── visualization/
│
├── results/
│   ├── real_only/
│   └── checkpoints/              # Local model weights; not committed
│
├── requirements.txt
├── .gitignore
└── README.md
```

This tree describes the intended project organization and the files currently identified during development. Verify the actual repository tree before treating every directory or file above as present.

## 4. Dataset: UCF101

The initial experiments use the UCF101 human action recognition dataset.

UCF101 contains videos representing 101 action classes, including activities such as walking, applying makeup, sports, and other everyday actions.

### Dataset details

| Property | Value |
|---|---|
| Dataset | UCF101 |
| Number of action classes | 101 |
| Video format | AVI |
| Training split | Official split 1 |
| Official training-list entries | 9,537 |
| Official test-list entries | 3,783 |
| Frames sampled per video | 16 |
| Input resolution | 112 × 112 |
| Image channels | 3 (RGB) |
| Tensor value range before model normalization | [0, 1] |

The dataset contains approximately 13,000 videos overall. Exact usable counts should be taken from the dataset files and loader output rather than assuming every listed video is present.

### Dataset files

- `data/UCF-101/` contains the downloaded video folders.
- `data/splits/classInd.txt` maps class names to class indices.
- `data/splits/trainlist01.txt` lists training videos for official split 1.
- `data/splits/testlist01.txt` lists test videos for official split 1.
- `trainlist02.txt` and `testlist02.txt` provide split 2.
- `trainlist03.txt` and `testlist03.txt` provide split 3.

The official split files are retained to support reproducible dataset selection. The current stratified experiments use split 1's training list and do not use the official test list for training or validation.

### Dataset preprocessing

The dataset loader in `analysis/dataset.py` performs the following operations:

1. Reads video paths and class labels.
2. Opens each video and decodes frames.
3. Samples 16 frames from each video.
4. Resizes frames to 112 × 112 pixels.
5. Converts frames to RGB tensors.
6. Scales pixel values to the [0, 1] range.
7. Returns the video tensor, numeric label, and class name.

The video tensor is initially represented as:

```text
[T, C, H, W]
```

where `T` is the number of frames, `C` is the number of channels, and `H` and `W` are image height and width.

For R3D-18, the batch is rearranged to:

```text
[B, C, T, H, W]
```

where `B` is the batch size.

The pretrained model's expected channel-wise normalization is then applied.

## 5. Models

### 5.1 Basic 3D CNN

A small 3D convolutional neural network was implemented to test the data-loading and classification pipeline.

Unlike a conventional 2D CNN, a 3D CNN applies convolution across spatial and temporal dimensions. This allows it to learn representations that capture both appearance and motion.

The initial small-data experiments produced low classification accuracy. These results helped verify the pipeline and motivated the use of a pretrained video-recognition model.

### 5.2 R3D-18 with transfer learning

The main baseline uses R3D-18, a 3D residual convolutional neural network provided by TorchVision.

The model starts with pretrained Kinetics-400 weights. Its final fully connected classification layer is replaced with a new layer containing 101 output classes for UCF101.

The current training setup freezes the pretrained backbone and trains only the classification head.

| Model property | Value |
|---|---|
| Architecture | R3D-18 |
| Initial weights | Kinetics-400 pretrained |
| Total parameters | 33,218,085 |
| Trainable parameters | 51,813 |
| Number of output classes | 101 |
| Training device | CPU |
| Framework | PyTorch and TorchVision |

This approach is a form of transfer learning. It reuses features learned from a large video dataset rather than training the entire network from scratch.

## 6. Training Experiments

The main experiment script is:

```text
analysis/train_r3d18_stratified.py
```

The script selects a fixed number of videos per action class from the official training split. For each class, 80% of the selected videos are assigned to training and 20% to validation.

With the current configurations, the split is class-balanced.

### Experiment configurations

| Configuration | Videos selected per class | Training videos | Validation videos | Epochs |
|---|---:|---:|---:|---:|
| Experiment 1 | 5 | 404 | 101 | 1 |
| Experiment 2 | 10 | 808 | 202 | 2 |
| Experiment 3 | 20 | 1,616 | 404 | 3 |

The selected-video totals include both training and validation examples. For example, selecting 20 videos per class gives 2,020 selected videos: 1,616 for training and 404 for validation.

All three experiments use seed 42. The model is trained with a batch size of 2 in the recorded runs.

### Experiment 1: Five videos per class

Command:

```powershell
python .\analysis\train_r3d18_stratified.py --samples-per-class 5 --epochs 1 --batch-size 2
```

Results:

| Metric | Result |
|---|---:|
| Training loss | 4.8724 |
| Training accuracy | 8.66% |
| Validation loss | 3.0448 |
| Validation accuracy | 32.67% |

This was the initial stratified experiment. The small sample size and one validation video per class limit the reliability of the validation estimate.

### Experiment 2: Ten videos per class

Command:

```powershell
python .\analysis\train_r3d18_stratified.py --samples-per-class 10 --epochs 2 --batch-size 2
```

Results:

| Metric | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Training loss | 3.7508 | 1.0148 |
| Training accuracy | 25.62% | 85.64% |
| Validation loss | 2.1159 | 1.2336 |
| Validation accuracy | 53.47% | 72.28% |

Increasing the number of examples per class improved the observed validation accuracy. However, the training and validation metrics should continue to be monitored for evidence of overfitting.

### Experiment 3: Twenty videos per class

Command:

```powershell
python .\analysis\train_r3d18_stratified.py --samples-per-class 20 --epochs 3 --batch-size 2
```

Results:

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| Training loss | 2.6853 | 0.5940 | 0.2733 |
| Training accuracy | 44.37% | 89.73% | 96.23% |
| Validation loss | 1.2573 | 0.7305 | 0.5736 |
| Validation accuracy | 70.05% | 85.64% | 86.39% |

The best validation accuracy recorded for this experiment was **86.39%**.

This is the strongest result among the three initial stratified experiments. Both training and validation loss decreased during training, while validation accuracy increased.

However, the result should be interpreted as preliminary. The validation set contains only four videos per class, and the official test set has not yet been evaluated.

### Overall comparison

| Experiment | Best validation accuracy |
|---|---:|
| 5 videos per class | 32.67% |
| 10 videos per class | 72.28% |
| 20 videos per class | 86.39% |

These experiments suggest that the pretrained classifier benefits from additional examples. They do not establish the final performance of the model on the full UCF101 dataset.

## 7. Outputs and Artifacts

The current training scripts save model weights and metrics locally.

### Checkpoints

```text
results/checkpoints/r3d18_best.pth
results/checkpoints/r3d18_stratified_best.pth
```

The first checkpoint is associated with the non-stratified R3D-18 training script, while the second is associated with the stratified training script.

The stratified script reuses the same checkpoint filename across runs. Therefore, the latest run may overwrite the previous checkpoint.

### Metrics

```text
results/real_only/r3d18_metrics.json
results/real_only/r3d18_stratified_metrics.json
```

The metrics JSON files record training and validation information from the corresponding scripts. Preserve separate copies when comparing configurations.

For example, the 20-videos-per-class run can be archived with:

```powershell
Copy-Item .\results\real_only\r3d18_stratified_metrics.json `
  .\results\real_only\r3d18_stratified_20_per_class_metrics.json

Copy-Item .\results\checkpoints\r3d18_stratified_best.pth `
  .\results\checkpoints\r3d18_stratified_20_per_class_best.pth
```

Check the output directory after running a script to confirm which files exist.

### Version-control policy

The UCF101 videos, virtual environment, and model checkpoints should remain outside Git. They are large files that should not be committed to the source repository.

The repository should track source code, configuration files, experiment descriptions, and appropriate lightweight evaluation results. Large artifacts can be stored separately if needed.

## 8. Environment and Setup

The project is currently being developed on Windows using PowerShell and a Python virtual environment.

The recorded experiments use CPU-only PyTorch because CUDA is unavailable in the current environment.

### Create and activate a virtual environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the appropriate execution policy permitted on your machine, or invoke the virtual environment's Python executable directly.

### Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the project's pinned dependency versions where available. The actual environment used for each experiment should be recorded, since library versions can affect reproducibility.

### Verify PyTorch

```powershell
python -c "import torch; import torchvision; print('PyTorch:', torch.__version__); print('TorchVision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Test the dataset loader

```powershell
python .\analysis\test_ucf101.py
```

### Test the classification model

```powershell
python .\analysis\test_model.py
```

These checks validate basic data loading and model execution. They do not constitute a complete performance evaluation.

## 9. Reproducing the Current Stratified Experiment

To reproduce the configuration with 20 videos selected per class:

```powershell
python .\analysis\train_r3d18_stratified.py --samples-per-class 20 --epochs 3 --batch-size 2
```

The script uses seed 42 by default. Its output includes the device, selected training and validation counts, epoch-level losses and accuracies, and the locations of saved metrics and checkpoints.

For a fair comparison, keep the dataset split, random seed, preprocessing, model initialization, and training configuration documented for each run.

Because the script overwrites standard output filenames, archive results before launching another experiment.

## 10. Research Methodology

The project follows a staged evaluation approach.

### Stage 1: Real-only baseline

Train a classifier using only real UCF101 videos. Establish reproducible training and validation metrics, then evaluate the selected model on the official held-out test split.

### Stage 2: Synthetic-video generation

Investigate suitable video-generation approaches and generate clips associated with the action classes or target domain. Record the generation model, prompts or conditioning information, frame rate, resolution, duration, and other generation settings.

This stage has not yet been completed.

### Stage 3: Synthetic-video quality evaluation

Evaluate whether generated clips preserve relevant visual content and temporal structure. Potential evaluation dimensions include:

- Visual quality and perceptual similarity.
- Temporal consistency between frames.
- Semantic correspondence between the generated clip and its intended class.
- Diversity across generated samples.
- Usefulness for downstream classification.

Visual realism alone is not sufficient evidence that a generated video is useful for machine learning.

### Stage 4: Real-plus-synthetic training

Train a comparable classifier using real videos augmented with synthetic videos. Control the number of training examples and other experimental variables so that the effect of synthetic augmentation can be measured.

### Stage 5: Final comparison

Evaluate the real-only and real-plus-synthetic models on the same held-out real test set.

Report the absolute change in accuracy, per-class performance, and relevant uncertainty or variability. A synthetic-data approach should only be considered beneficial if the evaluation provides evidence of improved generalization.

## 11. Current Limitations

The current experiments have several limitations:

- The training runs use only a subset of the UCF101 training videos.
- The validation set is small relative to the 101-class classification task.
- Only the final classification layer is trained; the pretrained backbone remains frozen.
- CPU training limits the practical number of experiments that can be performed.
- The official test split has not yet been used for final evaluation.
- Synthetic videos have not yet been incorporated into the training pipeline.
- No controlled real-only versus real-plus-synthetic comparison has been completed.

These limitations define the next steps rather than invalidate the current baseline work.

## 12. Next Steps

The planned development priorities are:

1. Implement an evaluation script for the official UCF101 test split.
2. Add top-5 accuracy, per-class metrics, and a confusion matrix.
3. Archive the current experiment results and document the training configuration.
4. Evaluate the selected real-only baseline on the held-out test set.
5. Investigate a suitable video-generation model and generation pipeline.
6. Develop synthetic-video quality and temporal-consistency evaluations.
7. Run controlled real-only versus real-plus-synthetic experiments.
8. Compare performance and determine whether synthetic data improves downstream classification.

## 13. Summary

The project currently has a functioning UCF101 video-loading pipeline, a basic 3D CNN, and a pretrained R3D-18 classifier trained using transfer learning. Initial stratified experiments reached 86.39% validation accuracy with 20 selected videos per class, 16 training videos per class, four validation videos per class, and three epochs.

This result provides a promising starting point for the real-only baseline, but final performance remains unverified on the official test set. The next major milestone is rigorous held-out evaluation, followed by synthetic-video generation and controlled experiments measuring its impact on downstream classification.
