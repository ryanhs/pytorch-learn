# Test 3: Transfer Learning — Fine-Tuning a Pretrained ResNet18

Building on **test2** (training a CNN from scratch), this test demonstrates **transfer learning** — the most important practical technique in modern deep learning.

Instead of training from zero, we take a model (**ResNet18**) that was already trained on **1.2 million ImageNet images** and adapt it to classify CIFAR-10 with minimal effort.

---

## The Core Idea

```
ImageNet (1.2M images, 1000 classes)
    │
    ▼
┌──────────────────────────────────────────────┐
│  ResNet18 Pretrained Backbone                │
│  (conv1 → layer1 → layer2 → layer3 → layer4)│
│  Already knows: edges, textures, shapes,     │
│  patterns, object parts                      │
│  ➜ FROZEN (requires_grad = False)            │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  New Classification Head                     │
│  Linear(512 → 10) for CIFAR-10              │
│  Starts with RANDOM weights                  │
│  ➜ TRAINABLE (the only part we train)        │
└──────────────────────────────────────────────┘
```

---

## What You Will See

| Phase | What Happens | Expected Accuracy |
| :--- | :--- | :--- |
| **Before** | Pretrained backbone + random untrained head | ~10% (random guessing) |
| **Phase 1** | Train only the new head (5 epochs) | ~88–92% |
| **Phase 2** | Unfreeze last conv block + gentle fine-tune (3 more epochs) | ~93–95% |
| *Reference* | test2 from-scratch CNN (10 epochs) | ~80% |

---

## Directory Structure

```
test3/
├── Dockerfile                          # Container with ROCm PyTorch + torchvision
├── run.sh                              # Build & run (with --network=host for model download)
├── README.md                           # This guide
└── docker_data/
    ├── train_transfer.py               # Full transfer learning pipeline
    ├── resnet18_cifar10_finetuned.pt    # Saved fine-tuned weights (after training)
    ├── comparison_transfer_learning.png # Before vs. After visual grid
    ├── training_curves_transfer.png     # Loss & accuracy over both phases
    └── parameter_efficiency.png         # Accuracy vs. parameters trained comparison
```

---

## How to Run

### Step 1: Launch the container
```bash
cd test3
./run.sh
```

### Step 2: Run the transfer learning pipeline
```bash
python train_transfer.py
```

The script will:
1. Download CIFAR-10 (~170 MB, first time only)
2. Download pretrained ResNet18 weights (~45 MB, first time only)
3. Show predictions **before** fine-tuning (random head = blind guessing)
4. Fine-tune in two phases on your AMD ROCm GPU
5. Show predictions **after** fine-tuning (accurate + confident)
6. Generate comparison plots

---

## Key PyTorch Concepts Introduced

### 1. Loading Pretrained Weights
```python
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
```
Downloads a model already trained on ImageNet. Its conv layers already detect visual features.

### 2. Freezing Layers (`requires_grad = False`)
```python
for param in model.parameters():
    param.requires_grad = False
```
Tells PyTorch: "Don't compute gradients for these layers. Don't update them during training." This locks the pretrained knowledge in place.

### 3. Replacing the Classification Head
```python
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(512, 10)    # 512 features → 10 CIFAR-10 classes
)
```
The original head mapped 512 features to 1000 ImageNet classes. We swap it for our 10-class task.

### 4. Differential Learning Rates
```python
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},   # Gentle for pretrained
    {'params': model.fc.parameters(),     'lr': 5e-4},   # Faster for new head
])
```
When unfreezing pretrained layers, use a smaller learning rate so you don't destroy the learned features. The new head can learn faster.

### 5. Two-Phase Training Strategy
- **Phase 1:** Only train the new head. Fast, safe, gets you most of the accuracy.
- **Phase 2:** Unfreeze the last conv block and fine-tune with low LR. Squeezes out the remaining accuracy by adapting deep features to your specific task.

