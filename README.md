# PyTorch & ROCm Learning Suite

A modular, Docker-based testing and learning workspace for **PyTorch** accelerated by **AMD ROCm GPUs** (supporting RDNA 3 / 3.5 / 4 architectures such as Radeon RX 9000 / 7000 series).

---

## Workspace Structure

```
pytorch-learn/
├── .gitignore                  # Clean ignore rules for datasets, checkpoints & logs
├── README.md                   # Repository overview (this file)
├── test1/                      # Test 1: Initial ROCm & YOLOv8 Exploration
│   ├── Dockerfile              # rocm/pytorch container with ultralytics
│   ├── run.sh                  # Interactive container launcher
│   ├── README.md               # Detailed notes for test 1
│   └── docker_data/            # Scripts (stress tests, YOLO demo, synthetic benchmarks)
└── test2/                      # Test 2: Real PyTorch CNN Learning (CIFAR-10)
    ├── Dockerfile              # rocm/pytorch container with torchvision & matplotlib
    ├── run.sh                  # Interactive container launcher (network enabled)
    ├── README.md               # Step-by-step guide to PyTorch fundamentals
    └── docker_data/            # Training pipeline & prediction tools
        ├── train_cifar10.py    # Custom CNN training with Before vs After evaluation
        └── predict.py          # Interactive model inference tool
```

---

## Prerequisites & GPU Acceleration

To use AMD GPU acceleration inside the Docker containers:
* **AMD GPU Drivers & ROCm** installed on the host.
* User permissions in `render` and `video` groups (`sudo usermod -aG render,video $USER`).
* Linux kernel access to `/dev/kfd` and `/dev/dri`.
* Modern RDNA cards use `HSA_OVERRIDE_GFX_VERSION=12.0.0` (already set inside the Dockerfiles).

---

## Quickstart

### Running Test 2 (Real PyTorch CNN Learning)
```bash
cd test2
./run.sh
```
Inside the container:
```bash
python train_cifar10.py
```
This trains a custom CNN on CIFAR-10, demonstrating:
- **Before training:** Random weights output ~10% accuracy (blind guessing).
- **After training:** ~80%+ accuracy with confident classifications.
- Generates `comparison_before_after.png` and `training_curves.png`.

---

### Running Test 1 (ROCm Verification & YOLOv8)
```bash
cd test1
./run.sh
```
Inside the container:
```bash
python test.py
```
