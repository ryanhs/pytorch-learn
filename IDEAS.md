# PyTorch & ROCm Learning Roadmap: Test 4 Ideation

This document tracks our completed learning milestones and outlines candidate architectures for **`test4`**.

---

## The Learning Progression So Far

| Stage | Focus | Architecture / Dataset | Key Takeaway | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Test 1** | **ROCm GPU Verification** | YOLOv8 / Synthetic Tensors | Verified AMD ROCm drivers, Docker pass-through (`/dev/kfd`, `/dev/dri`), and device performance. | **Completed [✓]** |
| **Test 2** | **PyTorch Foundations** | Custom CNN / CIFAR-10 | Built 5-step autograd loop from scratch (`nn.Module`, `DataLoader`, loss drop, ~10% $\rightarrow$ ~80% accuracy). | **Completed [✓]** |
| **Test 3** | **Transfer Learning** | Pretrained ResNet18 / CIFAR-10 | Model surgery: layer freezing (`requires_grad=False`), head swap, differential learning rates, ~95%+ accuracy in seconds. | **Completed [✓]** |
| **Test 4** | **"The Editor"** | Mini-UNet / Procedural Shapes | Dense pixel-level segmentation, skip connections (`torch.cat`), combined BCE+Dice loss, background cutout. | **Ready to Run [▶]** |
| **Test 5** | **"The Creator"** | Variational Autoencoder (VAE) | Generative synthesis, latent space exploration, reparameterization trick, morphing continuous distributions. | **Queued Next [⏳]** |

---

## Test 4 Candidates: Head-to-Head

The user has narrowed down the focus to two premier computer vision paradigms:

---

### Candidate A: Variational Autoencoder (VAE) — Generative Image Synthesis
* **The Paradigm:** Instead of predicting a single label from an image, the model learns the continuous probability distribution of images and **generates brand new images** from random numbers.
* **Architecture:**
  * **Encoder:** Compresses a 28×28 or 32×32 image into two vectors: Mean ($\mu$) and Log-Variance ($\log \sigma^2$).
  * **Latent Bottleneck:** Samples a vector $z$ using the **Reparameterization Trick**: $z = \mu + \epsilon \cdot \sigma$ (where $\epsilon \sim \mathcal{N}(0, I)$).
  * **Decoder:** Reconstructs the image from $z$ using transposed convolutions (`nn.ConvTranspose2d`).
* **Visual Deliverables / Outputs:**
  1. **Reconstruction Check:** Side-by-side plot of Original vs. Reconstructed images.
  2. **Generation from Pure Noise:** Feeding random coordinates $\mathcal{N}(0, 1)$ into the Decoder to synthesize never-before-seen images.
  3. **Latent Space Morph Grid:** A 2D grid smoothly morphing between classes (e.g., transforming a handwritten digit `3` continuously into an `8`).
* **Key PyTorch Concepts Taught:**
  * Multi-term custom loss: $\text{Total Loss} = \text{Reconstruction Loss (BCE/MSE)} + \text{KL Divergence}$.
  * The Reparameterization Trick (how backpropagation works through stochastic nodes).
  * Up-sampling with `nn.ConvTranspose2d`.

---

### Candidate B: Semantic Segmentation with a Mini-UNet — Pixel-Level Vision
* **The Paradigm:** Instead of whole-image classification ("this picture is a dog"), the model classifies **every single individual pixel** ("these pixels are the dog, these pixels are the background").
* **Architecture:**
  * **Encoder (Contracting Path):** Repeated convolutions and pooling to capture high-level context.
  * **Bottleneck:** Deepest compressed feature representation.
  * **Decoder (Expanding Path):** Transposed convolutions to recover spatial resolution.
  * **Skip Connections:** Concatenating encoder feature maps directly to decoder feature maps (`torch.cat([dec, skip], dim=1)`), preserving sharp high-resolution boundaries.
* **Visual Deliverables / Outputs:**
  1. **Before vs. After Overlay:** Untrained model outputs noisy static over the image; trained model outputs crisp pixel-accurate object masks.
  2. **Three-Panel Inspection:** Original Image $\rightarrow$ Ground Truth Mask $\rightarrow$ Predicted Segmentation Mask.
* **Key PyTorch Concepts Taught:**
  * Skip connections and tensor concatenation along channel dimensions.
  * Multi-dimensional loss computation (`[Batch, Classes, Height, Width]`).
  * Intersection-over-Union (IoU) and Dice coefficient evaluation metrics.

---

## Comparison Matrix

| Criteria | Candidate A: Variational Autoencoder (VAE) | Candidate B: Mini-UNet Segmentation |
| :--- | :--- | :--- |
| **Core Skill Learned** | Generative modeling & latent space representation | Dense pixel classification & skip connections |
| **Dataset Requirement** | MNIST / Fashion-MNIST (ultra-lightweight, built-in) | Oxford-IIIT Pet or Synthetic Shapes (lightweight masks) |
| **ROCm GPU Train Time** | ~30–45 seconds | ~45–60 seconds |
| **Visual Wow Factor** | High (generates new images + smooth latent morphing) | High (pixel-accurate outline masks overlaid on photos) |
| **Industry Usage** | Anomaly detection, image compression, diffusion foundation | Medical imaging, autonomous driving, background cutout |

---

## Decision Guide for Test 4

* **Choose VAE (Candidate A)** if you want to explore **Generative AI** — understanding how neural networks create new content and navigate mathematical latent spaces.
* **Choose UNet (Candidate B)** if you want to master **advanced discriminative vision** — moving from basic classification to pixel-level object extraction and skip-connection architectures.
