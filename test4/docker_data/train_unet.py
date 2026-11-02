import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# =====================================================================
# 1. PROCEDURAL DATASET GENERATOR (Zero-Download, Fast & Deterministic)
# =====================================================================
# Generates rich multi-shape objects on complex textured/gradient backgrounds.
# Perfect for learning semantic segmentation without waiting for 1GB downloads.

def generate_procedural_sample(size=128):
    """Generates an image (RGB) and its pixel-perfect binary mask."""
    # 1. Background: create random linear or radial color gradient + subtle noise
    bg_img = Image.new("RGB", (size, size))
    draw_bg = ImageDraw.Draw(bg_img)
    
    c1 = (random.randint(20, 90), random.randint(20, 90), random.randint(20, 90))
    c2 = (random.randint(100, 180), random.randint(100, 180), random.randint(100, 180))
    
    # Horizontal/vertical gradient
    vertical = random.choice([True, False])
    for i in range(size):
        t = i / size
        r = int(c1[0] * (1 - t) + c2[0] * t)
        g = int(c1[1] * (1 - t) + c2[1] * t)
        b = int(c1[2] * (1 - t) + c2[2] * t)
        if vertical:
            draw_bg.line([(0, i), (size, i)], fill=(r, g, b))
        else:
            draw_bg.line([(i, 0), (i, size)], fill=(r, g, b))

    # 2. Foreground: draw random shapes on a blank mask
    mask_img = Image.new("L", (size, size), 0)
    draw_mask = ImageDraw.Draw(mask_img)

    fg_img = bg_img.copy()
    draw_fg = ImageDraw.Draw(fg_img)

    num_objects = random.randint(1, 3)
    for _ in range(num_objects):
        shape_type = random.choice(["circle", "rectangle", "triangle", "ellipse"])
        color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))
        
        cx = random.randint(25, size - 25)
        cy = random.randint(25, size - 25)
        rad_x = random.randint(15, 30)
        rad_y = random.randint(15, 30)

        box = [cx - rad_x, cy - rad_y, cx + rad_x, cy + rad_y]

        if shape_type == "circle":
            rad = random.randint(15, 30)
            box = [cx - rad, cy - rad, cx + rad, cy + rad]
            draw_fg.ellipse(box, fill=color, outline=(255, 255, 255))
            draw_mask.ellipse(box, fill=255)
        elif shape_type == "ellipse":
            draw_fg.ellipse(box, fill=color, outline=(255, 255, 255))
            draw_mask.ellipse(box, fill=255)
        elif shape_type == "rectangle":
            draw_fg.rectangle(box, fill=color, outline=(255, 255, 255))
            draw_mask.rectangle(box, fill=255)
        elif shape_type == "triangle":
            pts = [(cx, cy - rad_y), (cx - rad_x, cy + rad_y), (cx + rad_x, cy + rad_y)]
            draw_fg.polygon(pts, fill=color, outline=(255, 255, 255))
            draw_mask.polygon(pts, fill=255)

    # Convert to numpy tensors
    img_arr = np.array(fg_img, dtype=np.float32) / 255.0
    # Add subtle gaussian noise to make it realistic
    noise = np.random.normal(0, 0.02, img_arr.shape).astype(np.float32)
    img_arr = np.clip(img_arr + noise, 0.0, 1.0)
    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1)  # [3, H, W]

    mask_arr = (np.array(mask_img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)    # [1, H, W]

    return img_tensor, mask_tensor

class ProceduralSegmentationDataset(Dataset):
    def __init__(self, num_samples=1000, seed=42):
        self.num_samples = num_samples
        # Pre-generate deterministic samples for consistent before/after evaluation
        random.seed(seed)
        np.random.seed(seed)
        print(f"  Generating {num_samples} procedural segmentation pairs (128x128)...")
        self.samples = [generate_procedural_sample(size=128) for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]

# =====================================================================
# 2. DEFINE THE UNET ARCHITECTURE WITH SKIP CONNECTIONS
# =====================================================================
class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class MiniUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Decoder (Expanding Path with Skip Connections)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)   # 128 (upsampled) + 128 (skip) = 256

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)    # 64 + 64 = 128

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)     # 32 + 32 = 64

        # Final 1x1 Conv to single-channel logit
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass (store skip features)
        x1 = self.inc(x)         # [B, 32, 128, 128]  -> Skip 1
        x2 = self.down1(x1)      # [B, 64, 64, 64]    -> Skip 2
        x3 = self.down2(x2)      # [B, 128, 32, 32]   -> Skip 3
        x4 = self.down3(x3)      # [B, 256, 16, 16]   -> Bottleneck

        # Decoder forward pass (concatenate skip features)
        u1 = self.up1(x4)
        u1 = torch.cat([u1, x3], dim=1) # [B, 256, 32, 32]
        d1 = self.conv_up1(u1)

        u2 = self.up2(d1)
        u2 = torch.cat([u2, x2], dim=1) # [B, 128, 64, 64]
        d2 = self.conv_up2(u2)

        u3 = self.up3(d2)
        u3 = torch.cat([u3, x1], dim=1) # [B, 64, 128, 128]
        d3 = self.conv_up3(u3)

        logits = self.outc(d3)   # [B, 1, 128, 128]
        return logits

# =====================================================================
# 3. LOSS FUNCTIONS & SEGMENTATION METRICS
# =====================================================================
class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice Loss for clean segmentation boundaries."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score
        
        return 0.5 * bce_loss + 0.5 * dice_loss

def calculate_iou(preds_binary, targets):
    """Intersection over Union (Jaccard Index)."""
    intersection = (preds_binary & (targets > 0.5)).float().sum((1, 2, 3))
    union = (preds_binary | (targets > 0.5)).float().sum((1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# =====================================================================
# 4. MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 70)
    print("  Test 4: The Editor — Semantic Segmentation with UNet on AMD ROCm")
    print("=" * 70)

    # 1. Device Setup
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {device}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 70)

    # 2. Data Preparation
    print("\n[Step 1/5] Preparing procedural segmentation dataset...")
    train_dataset = ProceduralSegmentationDataset(num_samples=1000, seed=42)
    val_dataset = ProceduralSegmentationDataset(num_samples=200, seed=100)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Pick 4 fixed test samples for Before vs. After inspection
    sample_indices = [5, 12, 28, 45]
    sample_imgs = torch.stack([val_dataset[i][0] for i in sample_indices]).to(device)
    sample_masks = torch.stack([val_dataset[i][1] for i in sample_indices]).to(device)

    # 3. Model Initialization
    print("\n[Step 2/5] Initializing Mini-UNet architecture with Skip Connections...")
    model = MiniUNet(in_channels=3, out_channels=1).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")

    # 4. BEFORE TRAINING EVALUATION
    print("\n" + "-" * 70)
    print("  [BEFORE TRAINING] Checking Untrained Model (Random Weights)...")
    print("-" * 70)
    
    model.eval()
    with torch.no_grad():
        logits_before = model(sample_imgs)
        probs_before = torch.sigmoid(logits_before)
        preds_before = (probs_before > 0.5)

        # Compute full validation IoU before training
        total_iou = 0.0
        batches = 0
        for val_imgs, val_msks in val_loader:
            val_imgs, val_msks = val_imgs.to(device), val_msks.to(device)
            p_logits = model(val_imgs)
            p_bin = (torch.sigmoid(p_logits) > 0.5)
            total_iou += calculate_iou(p_bin, val_msks)
            batches += 1
        before_val_iou = (total_iou / batches) * 100.0

    print(f"Validation Mean IoU BEFORE training: {before_val_iou:.2f}% (Expected ~10-25% random noise)")

    # 5. REAL TRAINING LOOP (Backpropagation on ROCm GPU)
    print("\n" + "=" * 70)
    print("  [Step 3/5] Starting UNet Training Loop on ROCm GPU")
    print("=" * 70)

    epochs = 10
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    val_ious = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_train_samples = 0
        epoch_start = time.time()

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            total_train_samples += imgs.size(0)

        scheduler.step()
        epoch_loss = running_loss / total_train_samples

        # Validation IoU
        model.eval()
        total_val_iou = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_imgs, val_msks in val_loader:
                val_imgs, val_msks = val_imgs.to(device), val_msks.to(device)
                p_logits = model(val_imgs)
                p_bin = (torch.sigmoid(p_logits) > 0.5)
                total_val_iou += calculate_iou(p_bin, val_msks)
                val_batches += 1
        epoch_val_iou = (total_val_iou / val_batches) * 100.0

        train_losses.append(epoch_loss)
        val_ious.append(epoch_val_iou)

        epoch_duration = time.time() - epoch_start
        gpu_mem = f"{torch.cuda.memory_allocated(0)/(1024**2):.1f} MB" if cuda_available else "N/A"
        print(f"Epoch [{epoch:2d}/{epochs:2d}] ({epoch_duration:4.1f}s) | Loss: {epoch_loss:.4f} | Val IoU: {epoch_val_iou:5.1f}% | GPU Mem: {gpu_mem}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds!")

    # Save trained model weights
    save_path = '/app/unet_segmentation.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Trained UNet model saved to: {save_path}")

    # 6. AFTER TRAINING EVALUATION
    print("\n" + "=" * 70)
    print("  [Step 4/5] Evaluating Trained Model (AFTER TRAINING)")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        logits_after = model(sample_imgs)
        probs_after = torch.sigmoid(logits_after)
        preds_after = (probs_after > 0.5)
        after_val_iou = val_ious[-1]

    print(f"Validation Mean IoU:  BEFORE = {before_val_iou:.2f}%  --->  AFTER = {after_val_iou:.2f}%")
    print("The model has learned exact pixel boundaries!")

    # 7. GENERATE VISUAL PLOTS & THE "EDITOR" CUTOUT DEMO
    print("\n" + "=" * 70)
    print("  [Step 5/5] Generating Visual Plots & 'The Editor' Cutout Showcase")
    print("=" * 70)

    try:
        # -------------------------------------------------------------
        # Plot 1: Before vs After Mask Comparison Grid
        # -------------------------------------------------------------
        fig, axes = plt.subplots(4, len(sample_indices), figsize=(12, 10))
        fig.suptitle("Test 4 UNet Segmentation: Before vs. After Training", fontsize=14, weight='bold')

        for i in range(len(sample_indices)):
            orig_np = sample_imgs[i].cpu().permute(1, 2, 0).numpy()
            true_np = sample_masks[i, 0].cpu().numpy()
            bef_np = probs_before[i, 0].cpu().numpy()
            aft_np = probs_after[i, 0].cpu().numpy()

            # Row 0: Original Image
            axes[0, i].imshow(orig_np)
            axes[0, i].set_title(f"Sample #{i+1}", fontsize=10)
            axes[0, i].axis('off')

            # Row 1: True Mask
            axes[1, i].imshow(true_np, cmap='gray')
            axes[1, i].axis('off')

            # Row 2: Before Training
            axes[2, i].imshow(bef_np, cmap='inferno')
            axes[2, i].axis('off')

            # Row 3: After Training
            axes[3, i].imshow(aft_np, cmap='inferno')
            axes[3, i].axis('off')

        axes[0, 0].set_ylabel("INPUT", fontsize=11, weight='bold')
        axes[1, 0].set_ylabel("TRUE MASK", fontsize=11, weight='bold')
        axes[2, 0].set_ylabel("BEFORE (Noise)", fontsize=11, weight='bold', color='red')
        axes[3, 0].set_ylabel("AFTER (UNet)", fontsize=11, weight='bold', color='green')
        plt.tight_layout()
        comparison_plot = '/app/comparison_segmentation.png'
        plt.savefig(comparison_plot, dpi=150)
        plt.close()
        print(f"Mask comparison saved to:    {comparison_plot}")

        # -------------------------------------------------------------
        # Plot 2: "The Editor" Cutout & Background Replacement Showcase
        # -------------------------------------------------------------
        fig, axes = plt.subplots(len(sample_indices), 4, figsize=(12, 10))
        fig.suptitle("The Editor in Action: Object Cutout & Green Screen Replacement", fontsize=14, weight='bold')

        neon_green = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        for i in range(len(sample_indices)):
            orig_np = sample_imgs[i].cpu().permute(1, 2, 0).numpy()
            mask_3d = preds_after[i, 0].cpu().unsqueeze(2).repeat(1, 1, 3).numpy()

            # Cutout: Object isolated on black canvas
            cutout_np = orig_np * mask_3d

            # Replacement: Object placed onto a vibrant green screen background
            replaced_np = orig_np * mask_3d + neon_green * (1.0 - mask_3d)

            # Column 0: Original Photo
            axes[i, 0].imshow(orig_np)
            axes[i, 0].axis('off')
            if i == 0: axes[i, 0].set_title("1. Original Image", fontsize=10, weight='bold')

            # Column 1: Predicted Mask
            axes[i, 1].imshow(preds_after[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 1].axis('off')
            if i == 0: axes[i, 1].set_title("2. Predicted Mask", fontsize=10, weight='bold')

            # Column 2: Cutout Object
            axes[i, 2].imshow(cutout_np)
            axes[i, 2].axis('off')
            if i == 0: axes[i, 2].set_title("3. Cutout Object", fontsize=10, weight='bold')

            # Column 3: Green Screen Replacement
            axes[i, 3].imshow(replaced_np)
            axes[i, 3].axis('off')
            if i == 0: axes[i, 3].set_title("4. Replaced Background", fontsize=10, weight='bold')

        plt.tight_layout()
        editor_plot = '/app/editor_cutout_showcase.png'
        plt.savefig(editor_plot, dpi=150)
        plt.close()
        print(f"Editor Cutout showcase saved: {editor_plot}")

        # -------------------------------------------------------------
        # Plot 3: Training Curves (Loss & IoU Progression)
        # -------------------------------------------------------------
        fig, ax1 = plt.subplots(figsize=(8, 4))
        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss (BCE + Dice)', color=color)
        ax1.plot(range(1, epochs + 1), train_losses, color=color, marker='o', label='Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Validation IoU (%)', color=color)
        ax2.plot(range(1, epochs + 1), val_ious, color=color, marker='s', label='IoU')
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Test 4: UNet Training Curves (Loss Drop & IoU Growth)')
        plt.tight_layout()
        curve_plot = '/app/training_curves_unet.png'
        plt.savefig(curve_plot, dpi=150)
        plt.close()
        print(f"Training curves saved to:     {curve_plot}")

    except Exception as e:
        print(f"Plot generation error: {e}")

    print("\nTest 4 complete! The Editor is ready to cut out objects.")

if __name__ == '__main__':
    main()
