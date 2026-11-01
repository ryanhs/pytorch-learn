# Test 3: Ideation & Roadmap

This document outlines candidate concepts and directions for **`test3`** as part of our PyTorch + AMD ROCm learning progression.

---

## The Learning Progression So Far

| Stage | Focus | Key Takeaway |
| :--- | :--- | :--- |
| **Test 1** | **ROCm GPU Verification** | Checked AMD ROCm drivers, Docker setup, device access (`/dev/kfd`, `/dev/dri`), and basic library wrappers (Ultralytics YOLOv8). |
| **Test 2** | **PyTorch Foundations from Scratch** | Real training loop on CIFAR-10 (`DataLoader`, `nn.Module`, 5-step autograd loop, before vs. after evaluation, visual loss curves). |
| **Test 3** | **Next Level (TBD)** | Moving from basic training from scratch into modern, practical, or generative deep learning paradigms. |

---

## Candidate Directions for Test 3

### 🥇 Option A: Transfer Learning & Fine-Tuning (Computer Vision)
* **The Concept:** In modern deep learning, models are rarely trained from scratch on raw pixels. Instead, engineers use **transfer learning**: starting from a model (like **ResNet-18**) pretrained on millions of images (ImageNet), freezing its feature extraction layers, and fine-tuning just the head on a specific target dataset.
* **Why It Fits:**
  * Direct evolutionary step from `test2`.
  * Demonstrates how a pretrained backbone reaches **95%+ accuracy in just 2–3 epochs**, easily outperforming the ~80% achieved from scratch.
  * Teaches how real-world vision models are deployed in production.
* **Key PyTorch Concepts Introduced:**
  * Loading pretrained weights (`torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`).
  * Freezing layer gradients (`param.requires_grad = False`).
  * Replacing final classification heads (`model.fc = nn.Linear(...)`).
  * Differential learning rates (small LR for backbone, higher LR for new head).

---

### 🥈 Option B: Text Generation with an RNN / Mini-Language Model (NLP)
* **The Concept:** Branching outside computer vision into natural language processing (NLP). The model learns to predict the next character in a text sequence (e.g., trained on Shakespeare or open-domain literature).
* **Why It Fits:**
  * Completely different domain that proves PyTorch handles arbitrary sequences, not just 2D images.
  * Very dramatic visual "Before vs. After":
    * **Before training:** Generates random, unreadable symbol soup (`xK#m2ä..`).
    * **After training:** Generates syntactically coherent words, punctuation, and dialogue structure.
* **Key PyTorch Concepts Introduced:**
  * Embedding layers (`nn.Embedding`).
  * Recurrent layers (`nn.LSTM` or `nn.GRU`) and sequence dimensions `(batch, seq_len, features)`.
  * Autoregressive text generation loop with temperature sampling.

---

### 🥉 Option C: Deep Convolutional GAN (Generative Modeling)
* **The Concept:** Training two networks against each other: a **Generator** that creates images out of random latent noise, and a **Discriminator** that tries to distinguish real images from fake ones.
* **Why It Fits:**
  * Fascinating generative behavior: the network creates new images that never previously existed.
  * Visceral progression: watching the generator evolve from television static to clean handwritten digits or shapes.
* **Key PyTorch Concepts Introduced:**
  * Dual-model alternating training loop (`optimizer_G` and `optimizer_D`).
  * Transposed convolutions (`nn.ConvTranspose2d`) for image upsampling.
  * Adversarial minimax loss functions (`nn.BCELoss`).

---

### 🏅 Option D: Denoising Autoencoder (Self-Supervised Reconstruction)
* **The Concept:** Taking clean images, corrupting them with severe synthetic noise or occlusion, and training an Encoder-Decoder network to reconstruct the original clean image.
* **Why It Fits:**
  * Clear side-by-side visual outputs (Noisy Input $\rightarrow$ Encoded Bottleneck $\rightarrow$ Reconstructed Output).
  * Demonstrates dimensionality reduction and latent space representation.
* **Key PyTorch Concepts Introduced:**
  * Encoder-decoder architecture (`Conv2d` downsampling $\rightarrow$ latent vector $\rightarrow$ `ConvTranspose2d` upsampling).
  * Pixel-wise reconstruction loss (`nn.MSELoss` or `nn.L1Loss`).

---

## Comparison Matrix

| Option | Practical Value | Fun / Wow Factor | ROCm GPU Speed | Recommended Next? |
| :--- | :---: | :---: | :---: | :---: |
| **A. Transfer Learning** | ⭐⭐⭐⭐⭐ (Highest) | ⭐⭐⭐ | ⚡ Ultra-fast (~30s) | **Yes (Top Pick)** |
| **B. Text Gen (RNN)** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚡ Fast (~1m) | Strong alternative |
| **C. DCGAN** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⏱️ Moderate (~3–5m) | Great for generative |
| **D. Autoencoder** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⚡ Fast (~1–2m) | Great for unsupervised |
