import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. DEFINE THE NEURAL NETWORK ARCHITECTURE
# =====================================================================
# A clean, modular Convolutional Neural Network (CNN) for 32x32 images.
class SimpleVisionCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Block 1: 3 channels (RGB) -> 32 filters, 32x32 -> 16x16
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Block 2: 32 -> 64 filters, 16x16 -> 8x8
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Block 3: 64 -> 128 filters, 8x8 -> 4x4
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Classifier Head: 128 * 4 * 4 = 2048 features -> 10 classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.classifier(x)
        return x

# =====================================================================
# HELPER: UN-NORMALIZE IMAGE FOR DISPLAY
# =====================================================================
def unnormalize(tensor_img):
    # Standard CIFAR-10 mean and std
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    img = tensor_img.clone().detach().cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def evaluate_accuracy(model, data_loader, device):
    """Calculates accuracy on a given dataset loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 70)
    print("  PyTorch Real Learning: CIFAR-10 Image Classifier on AMD ROCm GPU")
    print("=" * 70)

    # 1. Check Device (AMD ROCm / CUDA)
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device Selected: {device}")
    if cuda_available:
        print(f"ROCm GPU Device: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU VRAM:  {total_vram:.2f} GB")
    else:
        print("Note: Running on CPU.")
    print("=" * 70)

    # 2. Data Preparation (Transforms & DataLoaders)
    print("\n[Step 1/5] Loading CIFAR-10 Dataset (Real 32x32 color photos across 10 classes)...")
    
    # Normalization statistics for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    data_dir = '/app/data'
    os.makedirs(data_dir, exist_ok=True)

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"Classes: {', '.join(classes)}")
    print(f"Training samples: {len(train_set)} images")
    print(f"Test samples:     {len(test_set)} images")

    # 3. Model Initialization
    print("\n[Step 2/5] Initializing fresh CNN model with RANDOM weights...")
    model = SimpleVisionCNN(num_classes=10).to(device)

    # Select 6 fixed test samples to track "Before" vs "After"
    sample_indices = [3, 7, 10, 16, 25, 30]
    sample_images = torch.stack([test_set[i][0] for i in sample_indices]).to(device)
    sample_labels = [test_set[i][1] for i in sample_indices]

    # -------------------------------------------------------------
    # 4. BEFORE TRAINING INFERENCE CHECK
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  [BEFORE TRAINING] Running model with UNTRAINED random weights...")
    print("-" * 70)

    model.eval()
    before_test_acc = evaluate_accuracy(model, test_loader, device)
    print(f"Overall Test Accuracy BEFORE training: {before_test_acc:.2f}% (Expected ~10% for 10-class random guessing)")

    with torch.no_grad():
        outputs_before = model(sample_images)
        probs_before = torch.softmax(outputs_before, dim=1)
        preds_before = torch.argmax(probs_before, dim=1).cpu().numpy()
        confs_before = torch.max(probs_before, dim=1).values.cpu().numpy()

    print("\nSample predictions BEFORE training:")
    for i, idx in enumerate(sample_indices):
        true_cls = classes[sample_labels[i]]
        pred_cls = classes[preds_before[i]]
        status = "CORRECT" if pred_cls == true_cls else "WRONG"
        print(f"  Sample #{i+1} [True: {true_cls:>5}]: Predicted '{pred_cls:<5}' with {confs_before[i]*100:5.1f}% confidence  [{status}]")

    # -------------------------------------------------------------
    # 5. REAL TRAINING LOOP (Backpropagation on ROCm GPU)
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [Step 3/5] Starting REAL Training on AMD ROCm GPU")
    print("=" * 70)

    epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_accuracies = []

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            # Standard PyTorch Training Steps:
            # 1. Clear gradients from previous step
            optimizer.zero_grad()
            # 2. Forward pass (compute prediction)
            outputs = model(inputs)
            # 3. Compute loss
            loss = criterion(outputs, targets)
            # 4. Backward pass (autograd calculates gradients dLoss/dWeight)
            loss.backward()
            # 5. Optimizer step (updates weights using gradients)
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        scheduler.step()
        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / total_train
        train_acc = 100.0 * correct_train / total_train
        val_acc = evaluate_accuracy(model, test_loader, device)

        train_losses.append(epoch_loss)
        val_accuracies.append(val_acc)

        gpu_mem = f"{torch.cuda.memory_allocated(0)/(1024**2):.1f} MB" if cuda_available else "N/A"
        print(f"Epoch [{epoch:2d}/{epochs:2d}] ({epoch_time:4.1f}s) | Loss: {epoch_loss:.4f} | Train Acc: {train_acc:5.1f}% | Test Acc: {val_acc:5.1f}% | GPU Mem: {gpu_mem}")

    total_time = time.time() - start_train_time
    print(f"\nTraining completed in {total_time:.2f} seconds!")

    # Save trained model weights
    save_path = '/app/cifar10_cnn.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Trained model saved to: {save_path}")

    # -------------------------------------------------------------
    # 6. AFTER TRAINING INFERENCE CHECK
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [Step 4/5] Evaluating Trained Model (AFTER TRAINING)")
    print("=" * 70)

    model.eval()
    after_test_acc = evaluate_accuracy(model, test_loader, device)

    with torch.no_grad():
        outputs_after = model(sample_images)
        probs_after = torch.softmax(outputs_after, dim=1)
        preds_after = torch.argmax(probs_after, dim=1).cpu().numpy()
        confs_after = torch.max(probs_after, dim=1).values.cpu().numpy()

    # -------------------------------------------------------------
    # 7. COMPARISON TABLE & VISUALIZATION
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [Step 5/5] DIRECT COMPARISON: BEFORE vs AFTER")
    print("=" * 70)
    print(f"Overall Accuracy:  BEFORE = {before_test_acc:.2f}%  --->  AFTER = {after_test_acc:.2f}%\n")

    header = f"{'Sample':<8} | {'True Class':<10} | {'Before Guess (Conf)':<22} | {'After Guess (Conf)':<22} | {'Change'}"
    print(header)
    print("-" * len(header))

    for i in range(len(sample_indices)):
        true_c = classes[sample_labels[i]]
        b_c = classes[preds_before[i]]
        b_conf = confs_before[i] * 100
        a_c = classes[preds_after[i]]
        a_conf = confs_after[i] * 100

        before_str = f"{b_c} ({b_conf:4.1f}%)"
        after_str = f"{a_c} ({a_conf:4.1f}%)"
        
        status = "LEARNED! [✓]" if a_c == true_c else "STILL OFF [✗]"
        print(f"Image #{i+1:<2} | {true_c:<10} | {before_str:<22} | {after_str:<22} | {status}")

    print("=" * 70)

    # -------------------------------------------------------------
    # 8. GENERATE VISUAL PLOTS
    # -------------------------------------------------------------
    try:
        # Plot 1: Visual Side-by-Side Images
        fig, axes = plt.subplots(2, len(sample_indices), figsize=(15, 6))
        fig.suptitle("Model Predictions: Before Training (Random) vs. After Training (Learned)", fontsize=14, weight='bold')

        for i, idx in enumerate(sample_indices):
            raw_img = unnormalize(test_set[idx][0])
            true_c = classes[sample_labels[i]]

            # Row 0: Before Training
            axes[0, i].imshow(raw_img)
            b_c = classes[preds_before[i]]
            b_conf = confs_before[i] * 100
            color_before = 'green' if b_c == true_c else 'red'
            axes[0, i].set_title(f"BEFORE: {b_c}\n({b_conf:.1f}%)", color=color_before, fontsize=10)
            axes[0, i].axis('off')

            # Row 1: After Training
            axes[1, i].imshow(raw_img)
            a_c = classes[preds_after[i]]
            a_conf = confs_after[i] * 100
            color_after = 'green' if a_c == true_c else 'red'
            axes[1, i].set_title(f"AFTER: {a_c}\n({a_conf:.1f}%)\n[True: {true_c}]", color=color_after, fontsize=10)
            axes[1, i].axis('off')

        axes[0, 0].set_ylabel("BEFORE", fontsize=12, weight='bold')
        axes[1, 0].set_ylabel("AFTER", fontsize=12, weight='bold')
        plt.tight_layout()
        comparison_plot = '/app/comparison_before_after.png'
        plt.savefig(comparison_plot, dpi=150)
        plt.close()
        print(f"\nVisual comparison saved to: {comparison_plot}")

        # Plot 2: Training & Validation Curves
        fig, ax1 = plt.subplots(figsize=(8, 4))
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss', color=color)
        ax1.plot(range(1, epochs + 1), train_losses, color=color, marker='o', label='Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Test Accuracy (%)', color=color)
        ax2.plot(range(1, epochs + 1), val_accuracies, color=color, marker='s', label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Real Training Curves (Loss & Accuracy over Epochs)')
        plt.tight_layout()
        curve_plot = '/app/training_curves.png'
        plt.savefig(curve_plot, dpi=150)
        plt.close()
        print(f"Training curves saved to:    {curve_plot}")

    except Exception as e:
        print(f"Plot generation skipped or encountered error: {e}")

    print("\nDemo finished successfully! You now have a real trained PyTorch vision model on ROCm.")

if __name__ == '__main__':
    main()

