# Test 5: The Creator — Generative AI & Latent Space Exploration with VAE

Welcome to **Test 5: "The Creator"**!

In **tests 1–4**, all models were *discriminative*: given an image, predict a category or pixel mask.  
In **test 5**, our neural network becomes **a generative creator**: synthesizing brand new images from pure mathematics and random numbers.

Using a **Convolutional Variational Autoencoder (Conv-VAE)** with a continuous 2D latent space, we explore the mind of the neural network: mapping how it imagines, generates, and morphs visual reality.

---

## Architecture: Convolutional VAE

```
Input Image [28x28x1]
      │
      ▼
[Conv Encoder]  ──►  Linear  ──►  Mean vector (μ) & Log-variance vector (log σ²)
                                         │
                                         ▼ (Reparameterization: z = μ + ε · σ)
                                  Latent Vector z [2D]
                                         │
                                         ▼
                                  [Conv Decoder]  ──►  Synthesized Image [28x28x1]
```

### The Magic of the Reparameterization Trick
In standard autoencoders, points in the bottleneck are discrete, leaving vast "empty holes" where the decoder outputs garbled nonsense.  
In a **Variational Autoencoder (VAE)**:
1. We enforce the latent space to follow a standard Gaussian distribution $\mathcal{N}(0, I)$ using **Kullback-Leibler (KL) Divergence**.
2. We make random sampling differentiable using the **Reparameterization Trick**:
   $$z = \mu + \epsilon \odot \sigma \quad \text{where } \epsilon \sim \mathcal{N}(0, I)$$
3. As a result, the entire 2D space becomes a **smooth, continuous manifold**: every coordinate $(z_1, z_2)$ renders a valid, smoothly morphing digit!

---

## Directory Structure

```
test5/
├── Dockerfile                          # Container with ROCm PyTorch, torchvision & scipy
├── run.sh                              # Interactive container launcher with GPU access
├── README.md                           # This guide
└── docker_data/
    ├── train_vae.py                    # [Main] Conv-VAE training loop + generation pipeline
    ├── predict.py                      # [Tool] Interactive latent morph walk & synthesis
    ├── vae_creator.pt                  # Trained model weights (generated after training)
    ├── creator_morph_grid.png          # 20x20 continuous latent space manifold map
    ├── creator_random_generations.png  # Brand new creations from random noise
    ├── creator_reconstructions.png     # Original vs. Before vs. After training
    └── training_curves_vae.png         # Reconstruction BCE loss & KL divergence curves
```

---

## How to Run

### Step 1: Launch the Docker Container
```bash
cd test5
./run.sh
```

### Step 2: Train The Creator
Inside the container:
```bash
python train_vae.py
```
What you will see:
1. **Dataset:** Loads MNIST (28×28 digits).
2. **Before Training Check:** The untrained network outputs random blur/static.
3. **Training on AMD ROCm GPU:** 12 epochs of combined BCE + KL-Divergence loss (~25 seconds).
4. **The Creator Artifacts:**
   * `creator_morph_grid.png`: A 20×20 grid scanning the continuous 2D latent space $[-3, 3] \times [-3, 3]$. You will see digits smoothly morphing into one another!
   * `creator_random_generations.png`: 16 brand-new digits generated from standard normal noise $\mathcal{N}(0, 1)$.
   * `creator_reconstructions.png`: Side-by-side reconstruction quality.
   * `training_curves_vae.png`: Total loss, reconstruction loss, and KL divergence progression.

### Step 3: Run Interactive Latent Morph Walking
Interpolate smoothly between any two digits in 10 steps:
```bash
python predict.py
```

---

## Key PyTorch Concepts Introduced

1. **The Reparameterization Trick:**
   ```python
   std = torch.exp(0.5 * logvar)
   eps = torch.randn_like(std)
   z = mu + eps * std  # Differentiable sampling
   ```

2. **Combined VAE Loss:**
   ```python
   bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
   kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
   loss = bce + kld
   ```

3. **Latent Manifold Navigation:**
   Feeding continuous coordinates into `model.decode(z)` allows us to walk through the generative distribution.
