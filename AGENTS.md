# Agent Guidelines & Repository Policy

> **NOTICE FOR ALL AI AGENTS & CONTRIBUTORS**
> 
> This repository is **PUBLIC-INTENDED**. It is purely an open educational workspace and technical study ground for learning **PyTorch** with **AMD ROCm GPU acceleration**.

---

## 1. Public Domain & Zero-Secret Policy

* **Public Intent:** All code, documentation, scripts, and configurations in this repository are intended to be fully public and open-source.
* **No Secrets or Credentials:** **NEVER** commit, generate, or store private credentials, API keys, access tokens, SSH keys, passwords, or personal access identifiers anywhere in this repository.
* **No Confidential / Proprietary Data:** There is **no proprietary, private, enterprise, or confidential data** in this repository. All code is for personal study and open reference.
* **Public Datasets Only:** All training and testing workflows must rely exclusively on open, standard public datasets (such as CIFAR-10, ImageNet public samples, or built-in torchvision/ultralytics benchmarks).

---

## 2. Guidelines for AI Assistants & Autonomous Agents

When working in this repository:

1. **Maintain Clean Git Hygiene:**
   * Always respect [`.gitignore`](.gitignore).
   * Never track downloaded datasets (`**/data/`, `**/datasets/`), large binary checkpoints (`*.pt`, `*.bin`, `*.safetensors`), or transient training artifacts (`**/runs/`, `**/train_output/`).
2. **Preserve Hardware Portability:**
   * Scripts and Dockerfiles target AMD ROCm environments with `HSA_OVERRIDE_GFX_VERSION=12.0.0` (for RDNA 3 / 3.5 / 4 architectures). Keep device bindings (`/dev/kfd`, `/dev/dri`, `--network=host`, etc.) intact.
   * Prefer relative and dynamically resolved script paths (`$(dirname "$0")`) over hardcoded absolute host paths.
3. **Pedagogical Code Quality:**
   * Keep PyTorch code readable, clean, and educational.
   * Favor clear, idiomatic PyTorch workflows (`Dataset` / `DataLoader`, standard forward/backward passes, explicit training loops) over opaque wrappers when teaching fundamental concepts.
4. **Product Direction:**
   * Prefer experiments that advance the offline mobile food-label reader described in [`IDEAS.md`](IDEAS.md), rather than standalone toy models or a generic chatbot.
   * Favor appropriately licensed, open-weight models that can be evaluated and optimized for on-device inference. Keep deterministic, inspectable fallbacks for safety-relevant output, and never present ingredient matching as a guarantee that food is safe.
