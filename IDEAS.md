# PyTorch & ROCm Learning Roadmap: Product Direction

This document tracks completed learning milestones and the next step toward a useful mobile product.

---

## The Learning Progression So Far

| Stage | Focus | Architecture / Dataset | Key Takeaway | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Test 1** | **ROCm GPU Verification** | YOLOv8 / Synthetic Tensors | Verified AMD ROCm drivers, Docker pass-through (`/dev/kfd`, `/dev/dri`), and device performance. | **Completed [✓]** |
| **Test 2** | **PyTorch Foundations** | Custom CNN / CIFAR-10 | Built 5-step autograd loop from scratch (`nn.Module`, `DataLoader`, loss drop, ~10% $\rightarrow$ ~80% accuracy). | **Completed [✓]** |
| **Test 3** | **Transfer Learning** | Pretrained ResNet18 / CIFAR-10 | Model surgery: layer freezing (`requires_grad=False`), head swap, differential learning rates, ~95%+ accuracy in seconds. | **Completed [✓]** |
| **Test 4** | **"The Editor"** | Mini-UNet / Procedural Shapes | Dense pixel-level segmentation, skip connections (`torch.cat`), combined BCE+Dice loss, background cutout. | **Completed [✓]** |
| **Test 5** | **"The Creator"** | Variational Autoencoder (VAE) | Generative synthesis, latent space exploration, reparameterization trick, morphing continuous distributions. | **Completed [✓]** |

---

## Completed Architecture Comparison

Two computer vision paradigms were explored:

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

## Next Product: Offline Food-Label Reader

**Recommendation:** Build a mobile app that scans packaged-food labels, extracts the ingredient list, and highlights ingredients the user has chosen to watch for. Keep the scan and preferences on-device so the core feature works offline and private data does not need to leave the phone.

This is a better next step than another toy model or a generic chatbot: it solves one concrete problem, has a small and testable first release, and gives later model work a real product target.

### First usable version

1. Capture or import a label photo.
2. Run on-device OCR and show the recognized text so users can correct mistakes.
3. Let users maintain a list of ingredients to watch for; highlight exact matches and show the matching source text.
4. Handle poor or incomplete scans explicitly. Never claim a product is safe to eat; the app is an aid for reading labels, not medical or allergy advice.

Start with OCR plus transparent matching rules. This gives us a baseline to test before adding a language model and avoids using a model where deterministic code is easier to trust.

### Model path

When the baseline app works, evaluate small, open-weight text models for turning noisy OCR into structured ingredient lists. Fine-tune with a lightweight adapter on appropriately licensed, public data, then measure extraction accuracy, latency, memory, and package size on target phones. Quantize and export only after confirming the chosen model's license permits the intended use and its inference runtime supports the target devices.

Keep the model optional: users should still be able to inspect and correct OCR text, and the app should not silently turn an uncertain extraction into a confident safety claim.

### Why it fits this workspace

* It builds on the project's PyTorch and ROCm learning without making another benchmark the end goal.
* It creates a concrete path from experimentation to an Android/iOS prototype.
* It gives model improvements measurable outcomes: ingredient extraction quality and on-device resource use.
* It leaves room to compare optimized open models rather than committing prematurely to a large general-purpose model.

### Dataset for the first test

**Recommended source: [Open Food Facts](https://world.openfoodfacts.org/).** Its public product records include ingredient text (`ingredients_text`) and product image references. Its [AWS image dataset](https://openfoodfacts.github.io/openfoodfacts-server/api/aws-images-dataset/) also includes OCR results generated with Google Cloud Vision. That makes it a practical source for finding real package examples and building a small OCR/extraction evaluation set.

This is **not a ready-made, verified image-to-ingredients benchmark**: the ingredient text is contributor-entered, records can be incomplete or wrong, and a product's available images may not show its ingredients panel. For a first test:

1. Select a small pilot (around 100–300 products) with non-empty ingredient text and a clearly readable ingredients-panel photo. Manually check the pairing and transcription; exclude records where the panel is missing, unreadable, or in a language outside the test scope.
2. Keep the product barcode/ID with each example, and split train/validation/test by product (not by image or crop) to avoid leakage.
3. Use the printed ingredient panel as the OCR ground truth. Treat the database ingredient text as a comparison/reference, not unquestioned truth. Keep a separate set of phone-captured photos for the final on-device check.
4. Download only the selected examples outside the repository (for example under ignored `data/`); do not commit dataset images or exports.

The runnable starting point is [Test 6](test6/README.md): it currently seeds one English-language product so the image-download and manual-review flow can be checked before growing the pilot.

Useful official resources:

* [Data exports and reuse conditions](https://world.openfoodfacts.org/data) — exports include a full food CSV (currently documented as about 0.9 GB compressed / 9 GB uncompressed), so do not start by downloading the whole database.
* [Product data fields](https://static.openfoodfacts.org/data/data-fields.txt) — documents `ingredients_text` and image URL fields.
* [Image download guidance](https://openfoodfacts.github.io/openfoodfacts-server/api/how-to-download-images/) — recommends AWS-hosted image data for bulk downloads rather than retrieving many images one-by-one from the service.
* [AWS images dataset](https://openfoodfacts.github.io/openfoodfacts-server/api/aws-images-dataset/) — documents how to match image and OCR files to product barcodes; the bucket syncs monthly, so some recent images may be absent.
* [API documentation](https://openfoodfacts.github.io/openfoodfacts-server/api/) — for a small curated lookup, use the current API guidance and its request limits/custom User-Agent requirements.

**License and safety check:** Open Food Facts database reuse is subject to ODbL/DbCL terms, while product images are CC BY-SA and can include packaging artwork with other rights. Review the [reuse conditions](https://world.openfoodfacts.org/data) before redistributing data, images, or derived datasets/models; do not assume public availability means unrestricted use. Open Food Facts also warns that records may be inaccurate or incomplete and must not be used for medical purposes, so this dataset is for OCR/model evaluation—not to establish that a food is safe for someone to eat.
