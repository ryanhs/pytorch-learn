import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from train_vae import ConvVAE

def main():
    print("=" * 65)
    print("  Test 5: The Creator — Interactive Latent Morph & Synthesis Tool")
    print("=" * 65)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights_path = '/app/vae_creator.pt'
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found.")
        print("Please run `python train_vae.py` first to train the Creator VAE!")
        return

    # Load ConvVAE architecture & weights
    model = ConvVAE(latent_dim=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    print("Loaded trained Conv-VAE weights successfully.")

    # Load test set to pick interpolation endpoints
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.MNIST(root='/app/data', train=False, download=True, transform=transform)

    # 1. LATENT SPACE INTERPOLATION WALK (Morphing between two digits)
    print("\n[1/2] Generating a continuous latent walk between two digits...")
    # Pick two different digits
    idx1, idx2 = 3, 2  # Digit 1 (e.g. 0) and Digit 2 (e.g. 1)
    img1, label1 = test_set[idx1]
    img2, label2 = test_set[idx2]

    with torch.no_grad():
        mu1, _ = model.encode(img1.unsqueeze(0).to(device))
        mu2, _ = model.encode(img2.unsqueeze(0).to(device))

        num_steps = 10
        alpha_values = np.linspace(0.0, 1.0, num_steps)
        interpolated_imgs = []

        print(f"  Morphing: Digit {label1}  ===>  Digit {label2}")
        for t in alpha_values:
            z_interp = (1.0 - t) * mu1 + t * mu2
            decoded = model.decode(z_interp).cpu().squeeze().numpy()
            interpolated_imgs.append(decoded)

    # Plot the morph walk
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 2.5))
    fig.suptitle(f"Continuous Latent Morphing: Digit {label1} -> Digit {label2}", fontsize=13, weight='bold')

    for i in range(num_steps):
        axes[i].imshow(interpolated_imgs[i], cmap='magma')
        axes[i].set_title(f"t={alpha_values[i]:.1f}", fontsize=9)
        axes[i].axis('off')

    plt.tight_layout()
    morph_path = '/app/interactive_morph_walk.png'
    plt.savefig(morph_path, dpi=150)
    plt.close()
    print(f"Latent morph walk saved to: {morph_path}")

    # 2. INSTANT RANDOM SYNTHESIS
    print("\n[2/2] Synthesizing 5 brand-new digits from random noise...")
    with torch.no_grad():
        random_z = torch.randn(5, 2, device=device)
        synth_imgs = model.decode(random_z).cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, 5, figsize=(10, 2.5))
    fig.suptitle("Instant Creations from Random Noise", fontsize=12, weight='bold')
    for i in range(5):
        axes[i].imshow(synth_imgs[i], cmap='gray')
        axes[i].set_title(f"z=({random_z[i,0]:.1f}, {random_z[i,1]:.1f})", fontsize=8)
        axes[i].axis('off')

    plt.tight_layout()
    synth_path = '/app/interactive_random_creations.png'
    plt.savefig(synth_path, dpi=150)
    plt.close()
    print(f"Instant creations saved to:   {synth_path}")
    print("=" * 65)

if __name__ == '__main__':
    main()

