import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_unet import MiniUNet, generate_procedural_sample, calculate_iou

def main():
    print("=" * 60)
    print("  Test 4: The Editor — Interactive Cutout & Segmentation Tool")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights_path = '/app/unet_segmentation.pt'
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found.")
        print("Please run `python train_unet.py` first to train the UNet!")
        return

    # Load UNet architecture and weights
    model = MiniUNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    print("Loaded trained UNet weights successfully.")

    # Generate 3 brand new random samples
    print("\nGenerating 3 random test scenes with procedural objects...")
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    fig.suptitle("The Editor: Live Inference & Background Cutout", fontsize=14, weight='bold')

    neon_cyan = np.array([0.0, 0.9, 1.0], dtype=np.float32)

    with torch.no_grad():
        for i in range(3):
            img_tensor, mask_tensor = generate_procedural_sample(size=128)
            img_batch = img_tensor.unsqueeze(0).to(device)
            mask_batch = mask_tensor.unsqueeze(0).to(device)

            logits = model(img_batch)
            probs = torch.sigmoid(logits)
            pred_bin = (probs > 0.5)

            iou = calculate_iou(pred_bin, mask_batch) * 100.0

            orig_np = img_tensor.permute(1, 2, 0).numpy()
            mask_np = pred_bin[0, 0].cpu().numpy()
            mask_3d = pred_bin[0, 0].cpu().unsqueeze(2).repeat(1, 1, 3).numpy()

            cutout_np = orig_np * mask_3d
            replaced_np = orig_np * mask_3d + neon_cyan * (1.0 - mask_3d)

            print(f"  Scene #{i+1}: Cutout IoU Quality = {iou:5.1f}% | Foreground Pixels: {mask_np.sum():,}")

            # Plot columns
            axes[i, 0].imshow(orig_np)
            axes[i, 0].axis('off')
            if i == 0: axes[i, 0].set_title("Original Input", fontsize=10, weight='bold')

            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].axis('off')
            if i == 0: axes[i, 1].set_title(f"Predicted Mask", fontsize=10, weight='bold')

            axes[i, 2].imshow(cutout_np)
            axes[i, 2].axis('off')
            if i == 0: axes[i, 2].set_title("Object Cutout", fontsize=10, weight='bold')

            axes[i, 3].imshow(replaced_np)
            axes[i, 3].axis('off')
            if i == 0: axes[i, 3].set_title("Replaced Background", fontsize=10, weight='bold')

    output_path = '/app/predict_cutout_result.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nLive cutout results saved to: {output_path}")
    print("=" * 60)

if __name__ == '__main__':
    main()

