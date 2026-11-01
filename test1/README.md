# ROCm PyTorch & YOLOv8 Test Suite (`test1`)

This directory contains a self-contained Docker-based testing, benchmarking, and validation environment for **AMD ROCm PyTorch** and **Ultralytics YOLO** computer vision workflows.

It is configured to support modern AMD Radeon graphics architectures (including RDNA 4 / Navi 44 architectures such as the RX 9060 XT) using the `HSA_OVERRIDE_GFX_VERSION=12.0.0` environment variable.

---

## Directory Structure

```
test1/
├── Dockerfile                  # Container definition based on rocm/pytorch:latest
├── run.sh                      # Shell script to build and run the ROCm container
├── README.md                   # Documentation and test suite summary (this file)
└── docker_data/                # Mounted to /app inside the container
    ├── bus.jpg                 # Sample input image for inference tests
    ├── yolov8n.pt              # Base pre-trained YOLOv8 nano model
    │
    ├── test.py                 # [1] Basic GPU inference test
    ├── train.py                # [2] YOLOv8 nano mini-training & validation test
    ├── train_pytorch.py        # [3] Pure PyTorch ConvNet training benchmark
    ├── compare.py              # [4] Before vs. after fine-tuning inference comparison
    ├── demo_scratch_learn.py   # [5] Train-from-scratch demo (random init vs trained)
    ├── heavy_stress_test.py    # [6] Sustained GPU stress & FP16 AMP load test
    ├── train_heavy_yolo.py     # [7] Heavy YOLOv8 Medium (25.9M params) training
    │
    ├── datasets/               # Auto-downloaded datasets (e.g., coco8)
    ├── train_output/           # Training weights, metrics, and evaluation plots
    ├── compare_output/         # Visual comparison outputs (before vs after)
    └── output/                 # Inference test outputs
```

---

## Environment & Hardware Setup

- **Base Image:** `rocm/pytorch:latest`
- **Key Dependencies:** `ultralytics`, `opencv-python-headless`, `libgl1`, `libglib2.0-0`, `libxcb1`
- **ROCm GFX Override:** `HSA_OVERRIDE_GFX_VERSION=12.0.0`
- **Container Host Access:**
  - KFD Kernel Driver: `--device=/dev/kfd`
  - DRI Render Nodes: `--device=/dev/dri`
  - IPC & Shared Memory: `--ipc=host --shm-size=16g` (allows multi-worker PyTorch `DataLoader` without bottlenecks)
  - Volume Mount: `$(pwd)/docker_data:/app`

---

## Getting Started

### 1. Build and Run the Container
Execute `run.sh` from the host:

```bash
chmod +x run.sh
./run.sh
```

This builds the Docker image `rocmpytorch:test` and launches an interactive bash shell mapped to `/app`.

---

## Test Scripts Overview

All scripts are executed inside the Docker container (`/app`):

### 1. Basic Inference Sanity Test (`test.py`)
- **Purpose:** Verifies that PyTorch detects the AMD ROCm GPU device and executes basic inference.
- **Model:** Pre-trained `yolov8n.pt`
- **Execution:**
  ```bash
  python3 test.py
  ```
- **Output:** Annotated detection image saved to `/app/output/bus.jpg`.

---

### 2. YOLOv8 Mini Training Test (`train.py`)
- **Purpose:** Tests forward & backward propagation, loss calculation, optimizer steps, and GPU kernels during model training.
- **Dataset:** Ultralytics mini-benchmark `coco8.yaml` (8 images).
- **Settings:** 5 epochs, batch size 4, 4 workers, 640x640 image size.
- **Execution:**
  ```bash
  python3 train.py
  ```
- **Output:** Trained model weights (`best.pt`, `last.pt`), loss curves, confusion matrices, and metrics in `/app/train_output/`.

---

### 3. Pure PyTorch Benchmark (`train_pytorch.py`)
- **Purpose:** Evaluates raw PyTorch training performance and throughput without the Ultralytics abstraction.
- **Model:** Custom 3-block convolutional network (`SimpleConvNet`) with BatchNorm and Adaptive Average Pooling.
- **Workload:** 150 synthetic batches of 64 images (224x224), Adam optimizer, CrossEntropy loss.
- **Execution:**
  ```bash
  python3 train_pytorch.py
  ```
- **Output:** Prints step-by-step loss, allocated GPU VRAM, and total throughput (`images/sec`).

---

### 4. Inference Comparison: Before vs. After Training (`compare.py`)
- **Purpose:** Compares object detection predictions of the stock base model against the fine-tuned model.
- **Prerequisite:** Requires running `train.py` first to generate `train_output/weights/best.pt`.
- **Execution:**
  ```bash
  python3 compare.py
  ```
- **Output:** Terminal comparison table with detection confidence, plus visual images in `/app/compare_output/before/` and `/app/compare_output/after/`.

---

### 5. Learning from Scratch Demo (`demo_scratch_learn.py`)
- **Purpose:** Demonstrates neural network learning progression starting from randomly initialized weights (`yolov8n.yaml` instead of pre-trained weights).
- **Procedure:**
  1. Tests inference with an untrained, randomly initialized model (model detects nothing or random noise).
  2. Trains for 30 epochs on GPU using `coco8.yaml`.
  3. Re-evaluates inference with the trained weights to observe learned object detection capabilities.
- **Execution:**
  ```bash
  python3 demo_scratch_learn.py
  ```
- **Output:** Side-by-side verification saved in `/app/scratch_demo_output/`.

---

### 6. Sustained GPU Stress Test (`heavy_stress_test.py`)
- **Purpose:** Pushes GPU matrix compute units, memory bandwidth, and power consumption under continuous load.
- **Model:** Deep Residual Network (~30 million parameters) utilizing Automatic Mixed Precision (`torch.cuda.amp.autocast` and `GradScaler`).
- **Execution:**
  ```bash
  # Run for default 60 seconds
  python3 heavy_stress_test.py

  # Or specify custom duration (in seconds), e.g., 180s:
  python3 heavy_stress_test.py 180
  ```
- **Output:** Live progress reporting every 2 seconds: FPS, allocated/reserved VRAM, and average throughput.

---

### 7. Heavy YOLO Model Training (`train_heavy_yolo.py`)
- **Purpose:** Tests sustained training under a larger model footprint.
- **Model:** `yolov8m.pt` (YOLOv8 Medium, ~25.9 million parameters).
- **Settings:** 50 epochs on `coco8.yaml`, batch size 8.
- **Execution:**
  ```bash
  python3 train_heavy_yolo.py
  ```
- **Output:** Saved in `/app/heavy_train_output/`.

---

## Summary of Results & Verified Capabilities

| Component | Status | Verification Detail |
|---|---|---|
| **ROCm GPU Recognition** | Pass | Successfully detects AMD GPU via ROCm PyTorch (`cuda:0`) |
| **Inference (`test.py`)** | Pass | Ultralytics YOLOv8 inference completes on GPU |
| **Backpropagation & Gradients** | Pass | Full training loops functional in both pure PyTorch and Ultralytics |
| **Mixed Precision (FP16 AMP)** | Pass | `torch.cuda.amp` autocasting and gradient scaling fully supported |
| **Multi-worker Data Loading** | Pass | Shared memory configuration (`--ipc=host`, `--shm-size=16g`) prevents DataLoader deadlocks |
