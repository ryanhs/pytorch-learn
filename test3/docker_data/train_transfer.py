import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================
def unnormalize(tensor_img):
    """Reverse ImageNet normalization for display."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor_img.clone().detach().cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img, 0, 1)


def evaluate_accuracy(model, data_loader, device):
    """Calculate accuracy on a dataset."""
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


def count_parameters(model, only_trainable=False):
    """Count model parameters."""
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def predict_samples(model, sample_images, classes, device):
    """Get predictions + confidences for a batch of sample images."""
    model.eval()
    with torch.no_grad():
        outputs = model(sample_images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        confs = torch.max(probs, dim=1).values.cpu().numpy()
    return preds, confs


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 70)
    print("  Test 3: Transfer Learning — Pretrained ResNet18 on CIFAR-10")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1. DEVICE SETUP
    # -----------------------------------------------------------------
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {device}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 2. DATA PREPARATION
    # -----------------------------------------------------------------
    # ResNet18 expects 224x224 images with ImageNet normalization.
    # CIFAR-10 images are 32x32, so we resize them up.
    # This is a normal part of transfer learning — adapting your data
    # to match what the pretrained model was trained on.
    print("\n[Step 1/6] Loading CIFAR-10 with ImageNet-compatible transforms...")

    transform_train = transforms.Compose([
        transforms.Resize(224),                   # Scale 32x32 -> 224x224 for ResNet
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(                     # ImageNet mean & std (NOT CIFAR-10!)
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    data_dir = '/app/data'
    os.makedirs(data_dir, exist_ok=True)

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    batch_size = 64   # Smaller batch: 224x224 images use more VRAM than 32x32
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"Classes: {', '.join(classes)}")
    print(f"Training: {len(train_set)} images | Test: {len(test_set)} images")

    # Pick 8 fixed test samples to track throughout
    sample_indices = [3, 7, 10, 16, 25, 30, 42, 55]
    sample_images = torch.stack([test_set[i][0] for i in sample_indices]).to(device)
    sample_labels = [test_set[i][1] for i in sample_indices]

    # -----------------------------------------------------------------
    # 3. LOAD PRETRAINED ResNet18
    # -----------------------------------------------------------------
    # This is the core concept of transfer learning:
    #   ResNet18 was trained on ImageNet (1.2 million images, 1000 classes).
    #   Its convolutional layers already "know" how to detect edges, textures,
    #   shapes, and objects. We reuse all of that knowledge.
    print("\n[Step 2/6] Loading ResNet18 pretrained on ImageNet (1.2M images)...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Original output layer: fc -> {model.fc.in_features} inputs -> 1000 classes (ImageNet)")

    # -----------------------------------------------------------------
    # 4. SURGERY: FREEZE BACKBONE + REPLACE HEAD
    # -----------------------------------------------------------------
    # FREEZE: Set requires_grad=False on ALL existing parameters.
    #   This means the backward pass won't compute gradients for them,
    #   and the optimizer won't update their weights.
    #   The pretrained knowledge is locked in place.
    print("\n[Step 3/6] Performing model surgery...")
    print("  -> Freezing ALL pretrained backbone layers (no gradient updates)...")

    for param in model.parameters():
        param.requires_grad = False

    # REPLACE HEAD: Swap the final classification layer.
    #   The old head: Linear(512 -> 1000) for ImageNet's 1000 classes.
    #   Our new head: Linear(512 -> 10) for CIFAR-10's 10 classes.
    #   This new layer starts with RANDOM weights — it's the only part
    #   that needs to learn.
    num_features = model.fc.in_features   # 512 for ResNet18
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 10)
    )
    model = model.to(device)

    trainable_params = count_parameters(model, only_trainable=True)
    frozen_params = total_params - trainable_params
    print(f"  -> Replaced fc: Linear({num_features} -> 10) for CIFAR-10")
    print(f"  -> Frozen params:    {frozen_params:>10,}  (pretrained backbone — NOT updated)")
    print(f"  -> Trainable params: {trainable_params:>10,}  (new head — WILL be trained)")
    print(f"  -> Training only {trainable_params/total_params*100:.1f}% of the model!")

    # -----------------------------------------------------------------
    # 5. BEFORE FINE-TUNING: TEST WITH RANDOM HEAD
    # -----------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  [BEFORE FINE-TUNING] Random head on top of pretrained backbone...")
    print("-" * 70)

    before_acc = evaluate_accuracy(model, test_loader, device)
    preds_before, confs_before = predict_samples(model, sample_images, classes, device)

    print(f"Test Accuracy BEFORE fine-tuning: {before_acc:.2f}%")
    print("  (The backbone extracts great features, but the random head")
    print("   doesn't know how to map them to CIFAR-10 classes yet.)\n")

    print("Sample predictions BEFORE fine-tuning:")
    for i in range(len(sample_indices)):
        true_c = classes[sample_labels[i]]
        pred_c = classes[preds_before[i]]
        status = "✓" if pred_c == true_c else "✗"
        print(f"  [{status}] True: {true_c:<6} | Predicted: {pred_c:<6} ({confs_before[i]*100:5.1f}%)")

    # -----------------------------------------------------------------
    # 6. FINE-TUNING PHASE 1: TRAIN ONLY THE HEAD
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [Step 4/6] Fine-Tuning Phase 1: Training ONLY the new head")
    print("=" * 70)

    epochs_phase1 = 5
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients (the new head)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    train_losses = []
    val_accuracies = []
    start_time = time.time()

    for epoch in range(1, epochs_phase1 + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / total
        train_acc = 100.0 * correct / total
        val_acc = evaluate_accuracy(model, test_loader, device)

        train_losses.append(epoch_loss)
        val_accuracies.append(val_acc)

        gpu_mem = f"{torch.cuda.memory_allocated(0)/(1024**2):.1f} MB" if cuda_available else "N/A"
        print(f"  Epoch [{epoch}/{epochs_phase1}] ({epoch_time:4.1f}s) | Loss: {epoch_loss:.4f} | Train: {train_acc:5.1f}% | Test: {val_acc:5.1f}% | GPU: {gpu_mem}")

    phase1_time = time.time() - start_time
    phase1_acc = val_accuracies[-1]

    # -----------------------------------------------------------------
    # 7. FINE-TUNING PHASE 2: UNFREEZE LAST CONV BLOCK
    # -----------------------------------------------------------------
    # Now we go further: unfreeze the last residual block (layer4) so
    # the model can adapt its deepest feature detectors to CIFAR-10.
    # We use a LOWER learning rate to avoid destroying pretrained knowledge.
    print("\n" + "=" * 70)
    print("  [Step 5/6] Fine-Tuning Phase 2: Unfreezing last conv block (layer4)")
    print("=" * 70)

    print("  -> Unfreezing model.layer4 (last residual block)...")
    for param in model.layer4.parameters():
        param.requires_grad = True

    trainable_now = count_parameters(model, only_trainable=True)
    print(f"  -> Trainable params now: {trainable_now:,} (was {trainable_params:,})")

    # Differential learning rates:
    #   - layer4 (pretrained): small LR to gently adapt
    #   - fc head (our new layer): higher LR to keep learning
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},   # Gentle for pretrained
        {'params': model.fc.parameters(),     'lr': 5e-4},   # Faster for new head
    ])

    epochs_phase2 = 3

    for epoch in range(1, epochs_phase2 + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / total
        train_acc = 100.0 * correct / total
        val_acc = evaluate_accuracy(model, test_loader, device)

        train_losses.append(epoch_loss)
        val_accuracies.append(val_acc)

        total_epoch = epochs_phase1 + epoch
        gpu_mem = f"{torch.cuda.memory_allocated(0)/(1024**2):.1f} MB" if cuda_available else "N/A"
        print(f"  Epoch [{total_epoch}/{epochs_phase1+epochs_phase2}] ({epoch_time:4.1f}s) | Loss: {epoch_loss:.4f} | Train: {train_acc:5.1f}% | Test: {val_acc:5.1f}% | GPU: {gpu_mem}")

    total_time = time.time() - start_time
    final_acc = val_accuracies[-1]

    # Save the fine-tuned model
    save_path = '/app/resnet18_cifar10_finetuned.pt'
    torch.save(model.state_dict(), save_path)
    print(f"\nFine-tuned model saved to: {save_path}")

    # -----------------------------------------------------------------
    # 8. AFTER FINE-TUNING: COMPARE
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [Step 6/6] RESULTS: Before vs. After Fine-Tuning")
    print("=" * 70)

    preds_after, confs_after = predict_samples(model, sample_images, classes, device)

    print(f"\nOverall Test Accuracy:")
    print(f"  BEFORE fine-tuning (random head):  {before_acc:5.2f}%")
    print(f"  AFTER Phase 1 (head only):         {phase1_acc:5.2f}%")
    print(f"  AFTER Phase 2 (head + layer4):     {final_acc:5.2f}%")
    print(f"  [Reference: test2 from-scratch CNN: ~80%]")
    print(f"\nTotal fine-tuning time: {total_time:.1f}s")

    # Sample comparison table
    print(f"\n{'Sample':<8} | {'True':<7} | {'Before (Conf)':<18} | {'After (Conf)':<18} | {'Result'}")
    print("-" * 80)
    for i in range(len(sample_indices)):
        true_c = classes[sample_labels[i]]
        b_c = classes[preds_before[i]]
        a_c = classes[preds_after[i]]
        b_str = f"{b_c} ({confs_before[i]*100:4.1f}%)"
        a_str = f"{a_c} ({confs_after[i]*100:4.1f}%)"

        if a_c == true_c and b_c != true_c:
            status = "LEARNED ✓"
        elif a_c == true_c and b_c == true_c:
            status = "STILL OK ✓"
        else:
            status = "MISSED  ✗"
        print(f"#{i+1:<6} | {true_c:<7} | {b_str:<18} | {a_str:<18} | {status}")

    # -----------------------------------------------------------------
    # KEY INSIGHT: WHY TRANSFER LEARNING WORKS
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  KEY INSIGHT: Why Transfer Learning is Powerful")
    print("=" * 70)
    print(f"""
  In test2, we trained a CNN from SCRATCH on CIFAR-10:
    - Had to learn everything (edges, textures, shapes, objects)
    - 10 epochs, reached ~80% accuracy

  In test3, we used a PRETRAINED ResNet18:
    - The backbone already knew visual features from 1.2M ImageNet images
    - We only trained {trainable_params:,} new parameters (the head)
    - Then gently adapted {trainable_now:,} parameters (head + last block)
    - {epochs_phase1+epochs_phase2} epochs total, reached {final_acc:.1f}% accuracy

  Transfer learning = standing on the shoulders of giants.
  You don't need millions of images or days of training.
  You reuse what someone else already learned.
""")

    # -----------------------------------------------------------------
    # 9. GENERATE PLOTS
    # -----------------------------------------------------------------
    try:
        total_epochs = epochs_phase1 + epochs_phase2
        epoch_range = range(1, total_epochs + 1)

        # Plot 1: Before vs After sample predictions
        fig, axes = plt.subplots(2, len(sample_indices), figsize=(16, 6))
        fig.suptitle("Transfer Learning: Before Fine-Tuning vs. After Fine-Tuning", fontsize=14, weight='bold')

        for i, idx in enumerate(sample_indices):
            raw_img = unnormalize(test_set[idx][0])
            true_c = classes[sample_labels[i]]

            # Row 0: Before
            axes[0, i].imshow(raw_img)
            b_c = classes[preds_before[i]]
            color = 'green' if b_c == true_c else 'red'
            axes[0, i].set_title(f"{b_c}\n({confs_before[i]*100:.0f}%)", color=color, fontsize=9)
            axes[0, i].axis('off')

            # Row 1: After
            axes[1, i].imshow(raw_img)
            a_c = classes[preds_after[i]]
            color = 'green' if a_c == true_c else 'red'
            axes[1, i].set_title(f"{a_c}\n({confs_after[i]*100:.0f}%)\n[True: {true_c}]", color=color, fontsize=9)
            axes[1, i].axis('off')

        axes[0, 0].set_ylabel("BEFORE", fontsize=12, weight='bold')
        axes[1, 0].set_ylabel("AFTER", fontsize=12, weight='bold')
        plt.tight_layout()
        plt.savefig('/app/comparison_transfer_learning.png', dpi=150)
        plt.close()
        print("Visual comparison saved to: /app/comparison_transfer_learning.png")

        # Plot 2: Training curves with phase annotations
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color_loss = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss', color=color_loss)
        ax1.plot(epoch_range, train_losses, color=color_loss, marker='o', label='Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax2 = ax1.twinx()
        color_acc = 'tab:blue'
        ax2.set_ylabel('Test Accuracy (%)', color=color_acc)
        ax2.plot(epoch_range, val_accuracies, color=color_acc, marker='s', label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)

        # Mark the phase boundary
        ax1.axvline(x=epochs_phase1 + 0.5, color='gray', linestyle='--', alpha=0.7)
        ax1.text(epochs_phase1 / 2, max(train_losses) * 0.9, 'Phase 1:\nHead Only',
                 ha='center', fontsize=10, color='gray')
        ax1.text(epochs_phase1 + (epochs_phase2 / 2) + 0.5, max(train_losses) * 0.9, 'Phase 2:\n+ Layer4',
                 ha='center', fontsize=10, color='gray')

        plt.title('Transfer Learning: Training Curves (Two-Phase Fine-Tuning)')
        plt.tight_layout()
        plt.savefig('/app/training_curves_transfer.png', dpi=150)
        plt.close()
        print("Training curves saved to:    /app/training_curves_transfer.png")

        # Plot 3: Parameter efficiency comparison
        fig, ax = plt.subplots(figsize=(8, 4))
        categories = ['From Scratch\n(test2 CNN)', 'Transfer Phase 1\n(Head Only)', 'Transfer Phase 2\n(Head + Layer4)']
        trained_counts = [
            338_000,          # Approximate from test2's SimpleVisionCNN
            trainable_params,
            trainable_now
        ]
        accuracies = [80.0, phase1_acc, final_acc]
        colors = ['#ff7f0e', '#2ca02c', '#1f77b4']

        bars = ax.bar(categories, accuracies, color=colors, edgecolor='black', linewidth=0.5)
        for bar, count, acc in zip(bars, trained_counts, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{acc:.1f}%\n({count:,} params)', ha='center', fontsize=9)

        ax.set_ylabel('Test Accuracy (%)')
        ax.set_ylim(0, 105)
        ax.set_title('Accuracy vs. Parameters Trained')
        plt.tight_layout()
        plt.savefig('/app/parameter_efficiency.png', dpi=150)
        plt.close()
        print("Parameter comparison saved:  /app/parameter_efficiency.png")

    except Exception as e:
        print(f"Plot generation error: {e}")

    print("\nTest 3 complete!")


if __name__ == '__main__':
    main()

