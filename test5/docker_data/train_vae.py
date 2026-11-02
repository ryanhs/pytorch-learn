import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# =====================================================================
# 1. CONVOLUTIONAL VARIATIONAL AUTOENCODER (Conv-VAE)
# =====================================================================
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x28x28 -> Latent distribution (mu, logvar)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 14, 14]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [B, 64, 7, 7]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True)
        )
        # Latent space heads: Mean (mu) and Log-Variance (logvar)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder: Latent vector z -> 1x28x28 reconstructed image
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64 * 7 * 7),
            nn.ReLU(inplace=True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # [B, 32, 14, 14]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 1, 28, 28]
            nn.Sigmoid()  # Pixel values bounded in [0, 1]
        )

    def encode(self, x):
        h = self.encoder_conv(x)
        h = self.encoder_fc(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        The Reparameterization Trick:
        Allows backpropagation through stochastic nodes by shifting
        randomness to an independent noise term epsilon ~ N(0, I).
        z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 64, 7, 7)
        return self.decoder_conv(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# =====================================================================
# 2. VAE LOSS: RECONSTRUCTION + KL-DIVERGENCE
# =====================================================================
def vae_loss_function(recon_x, x, mu, logvar):
    """
    Total Loss = Reconstruction Loss (BCE) + Regularization (KL Divergence).
    - Reconstruction Loss forces decoder to output sharp, recognizable images.
    - KL Divergence pulls the latent distribution toward standard normal N(0, I),
      preventing clusters from drifting apart and making the latent space continuous.
    """
    # 1. Pixel-wise binary cross entropy
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # 2. Kullback-Leibler Divergence: D_KL( q(z|x) || p(z) )
    # Analytical formula for two Gaussians: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return bce + kld, bce, kld

# =====================================================================
# 3. MAIN PIPELINE
# =====================================================================
def main():
    print("=" * 70)
    print("  Test 5: The Creator — Generative Conv-VAE on AMD ROCm GPU")
    print("=" * 70)

    # 1. Device Setup
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {device}")
    if cuda_available:
        print(f"ROCm GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM: {vram:.2f} GB")
    print("=" * 70)

    # 2. Data Preparation
    print("\n[Step 1/5] Loading MNIST dataset (Clean 28x28 handwritten digits)...")
    data_dir = '/app/data'
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor()  # Values in [0, 1]
    ])

    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training images: {len(train_set):,}")
    print(f"Test images:     {len(test_set):,}")

    # Pick 8 sample images to evaluate reconstruction
    sample_indices = [0, 1, 2, 3, 4, 5, 6, 7]
    sample_imgs = torch.stack([test_set[i][0] for i in sample_indices]).to(device)

    # 3. Model Initialization
    latent_dim = 2  # 2D continuous latent manifold for 2D visual grid exploration
    print(f"\n[Step 2/5] Initializing Conv-VAE with a {latent_dim}D continuous latent space...")
    model = ConvVAE(latent_dim=latent_dim).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")

    # 4. BEFORE TRAINING CHECK
    print("\n" + "-" * 70)
    print("  [BEFORE TRAINING] Checking Untrained Model (Random Weights)...")
    print("-" * 70)

    model.eval()
    with torch.no_grad():
        recon_before, _, _ = model(sample_imgs)
        # Generate from random noise before training
        random_noise = torch.randn(16, latent_dim, device=device)
        generated_before = model.decode(random_noise)

    print("Untrained model generates pure random static / blur.")

    # 5. REAL TRAINING LOOP (Backpropagation on ROCm GPU)
    print("\n" + "=" * 70)
    print("  [Step 3/5] Starting Generative VAE Training Loop on ROCm GPU")
    print("=" * 70)

    epochs = 12
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses = []
    bce_losses = []
    kld_losses = []

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        total_bce = 0.0
        total_kld = 0.0
        total_samples = 0
        epoch_start = time.time()

        for imgs, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)

            optimizer.zero_grad()
            recon_imgs, mu, logvar = model(imgs)
            loss, bce, kld = vae_loss_function(recon_imgs, imgs, mu, logvar)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()
            total_samples += imgs.size(0)

        scheduler.step()
        epoch_duration = time.time() - epoch_start
        avg_loss = total_epoch_loss / total_samples
        avg_bce = total_bce / total_samples
        avg_kld = total_kld / total_samples

        train_losses.append(avg_loss)
        bce_losses.append(avg_bce)
        kld_losses.append(avg_kld)

        gpu_mem = f"{torch.cuda.memory_allocated(0)/(1024**2):.1f} MB" if cuda_available else "N/A"
        print(f"Epoch [{epoch:2d}/{epochs:2d}] ({epoch_duration:4.1f}s) | Loss: {avg_loss:6.2f} (BCE: {avg_bce:6.2f}, KLD: {avg_kld:5.2f}) | GPU: {gpu_mem}")

    total_time = time.time() - start_train_time
    print(f"\nTraining completed in {total_time:.2f} seconds!")

    # Save trained model weights
    save_path = '/app/vae_creator.pt'
    torch.save(model.state_dict(), save_path)
    print(f"Trained VAE weights saved to: {save_path}")

    # 6. AFTER TRAINING GENERATION & EVALUATION
    print("\n" + "=" * 70)
    print("  [Step 4/5] Evaluating Trained Creator (AFTER TRAINING)")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        recon_after, _, _ = model(sample_imgs)
        # Generate brand new images from random normal noise N(0, 1)
        generated_after = model.decode(random_noise)

    # 7. GENERATE VISUAL CREATIVE ARTIFACTS
    print("\n" + "=" * 70)
    print("  [Step 5/5] Generating Visual Creator Artifacts & Latent Morph Grid")
    print("=" * 70)

    try:
        # -------------------------------------------------------------
        # Artifact 1: Continuous 2D Latent Space Morph Grid (The Creator's Mind)
        # -------------------------------------------------------------
        # We sample a 20x20 grid in the Gaussian inverse CDF space [-3, 3]
        n = 20
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
        
        figure = np.zeros((28 * n, 28 * n))

        with torch.no_grad():
            for i, yi in enumerate(grid_x):
                for j, xi in enumerate(grid_y):
                    z_sample = torch.tensor([[xi, yi]], dtype=torch.float32, device=device)
                    x_decoded = model.decode(z_sample).cpu().squeeze().numpy()
                    figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = x_decoded

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='gnuplot2')
        plt.title("The Creator's Mind: Continuous 2D Latent Space Morph Grid", fontsize=14, weight='bold')
        plt.axis('off')
        plt.tight_layout()
        morph_plot = '/app/creator_morph_grid.png'
        plt.savefig(morph_plot, dpi=150)
        plt.close()
        print(f"1. Continuous latent morph grid saved to:    {morph_plot}")

        # -------------------------------------------------------------
        # Artifact 2: Brand New Image Synthesis from Pure Noise N(0, 1)
        # -------------------------------------------------------------
        fig, axes = plt.subplots(4, 4, figsize=(7, 7))
        fig.suptitle("Synthesized from Pure Random Noise N(0, 1)", fontsize=13, weight='bold')

        for idx, ax in enumerate(axes.flat):
            img_np = generated_after[idx, 0].cpu().numpy()
            ax.imshow(img_np, cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        synth_plot = '/app/creator_random_generations.png'
        plt.savefig(synth_plot, dpi=150)
        plt.close()
        print(f"2. Random noise synthesized creations saved:  {synth_plot}")

        # -------------------------------------------------------------
        # Artifact 3: Reconstruction Check (Original vs Before vs After)
        # -------------------------------------------------------------
        fig, axes = plt.subplots(3, len(sample_indices), figsize=(14, 5))
        fig.suptitle("Reconstruction Check: Original vs. Before Training vs. After Training", fontsize=13, weight='bold')

        for i in range(len(sample_indices)):
            orig_np = sample_imgs[i, 0].cpu().numpy()
            bef_np = recon_before[i, 0].cpu().numpy()
            aft_np = recon_after[i, 0].cpu().numpy()

            axes[0, i].imshow(orig_np, cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f"Digit #{test_set[sample_indices[i]][1]}", fontsize=9)

            axes[1, i].imshow(bef_np, cmap='inferno')
            axes[1, i].axis('off')

            axes[2, i].imshow(aft_np, cmap='gray')
            axes[2, i].axis('off')

        axes[0, 0].set_ylabel("ORIGINAL", fontsize=10, weight='bold')
        axes[1, 0].set_ylabel("BEFORE (Noise)", fontsize=10, weight='bold', color='red')
        axes[2, 0].set_ylabel("AFTER (Recon)", fontsize=10, weight='bold', color='green')

        plt.tight_layout()
        recon_plot = '/app/creator_reconstructions.png'
        plt.savefig(recon_plot, dpi=150)
        plt.close()
        print(f"3. Reconstruction comparison saved to:        {recon_plot}")

        # -------------------------------------------------------------
        # Artifact 4: Training Curves (Loss, BCE, and KLD)
        # -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.plot(range(1, epochs + 1), train_losses, color='tab:red', marker='o', label='Total Loss')
        ax1.plot(range(1, epochs + 1), bce_losses, color='tab:orange', linestyle='--', label='BCE (Reconstruction)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss per Sample')
        ax1.set_title('Reconstruction Loss Descent')
        ax1.legend()

        ax2.plot(range(1, epochs + 1), kld_losses, color='tab:blue', marker='s', label='KL Divergence')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('KL Divergence')
        ax2.set_title('Latent Space Regularization (KLD)')
        ax2.legend()

        plt.tight_layout()
        curves_plot = '/app/training_curves_vae.png'
        plt.savefig(curves_plot, dpi=150)
        plt.close()
        print(f"4. VAE training curves saved to:              {curves_plot}")

    except Exception as e:
        print(f"Plot generation error: {e}")

    print("\nTest 5 complete! The Creator is ready to synthesize images.")

if __name__ == '__main__':
    main()
