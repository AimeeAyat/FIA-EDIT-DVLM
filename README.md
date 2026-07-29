
<div align="center">

# **Flux & Key**
### Reference-Guided Image Composition with K/V Injection and Domain Guidance

*Built upon the EEdit editing framework with attention-level reference conditioning for FLUX.1*

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-yellow.svg)]()
[![FLUX.1](https://img.shields.io/badge/Model-FLUX.1-success.svg)]()

</div>

---

# Overview

**Flux & Key** is a reference-guided image composition framework built on top of the **EEdit** backbone. Rather than reproducing FIA or the original K/V pipeline, this project extends EEdit with **reference K/V attention injection**, **domain-aware prompt augmentation**, **tail classifier-free guidance (CFG)** and **background anchoring** for improved identity preservation during cross-domain image composition.

Our method addresses the common limitation of prompt-only image editing where semantic attributes are transferred but object identity gradually drifts during denoising.

---

#  Features

- Reference-guided image composition using FLUX.1
- Built upon the EEdit editing pipeline
- Reference Key/Value injection into early FLUX attention blocks
- Domain-aware prompt augmentation
- Tail CFG guidance
- Background anchoring
- Extensive qualitative comparisons
- Quantitative evaluation on TF-ICON
- Ablation studies and configuration analysis

---

#  Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/rabiaaslam92/FIA-EDIT-DVLM/main/assets/architecture.png" width="100%">
</p>

The architecture consists of:

1. Input & latent composition
2. Inherited EEdit backbone (RF-Inversion, Step Skipping, SLoC)
3. Proposed reference-guided extensions
   - Reference forward pass
   - Reference K/V memory
   - Reference attention injection
   - Domain prompt augmentation
   - Tail CFG
4. Modified FLUX denoising loop
5. VAE decoding
6. Final composed output

---

#  Qualitative Results



<p align="center">
  <img src="https://raw.githubusercontent.com/rabiaaslam92/FIA-EDIT-DVLM/main/assets/qualitative_results.png" width="100%">
</p>


#  Quantitative Results

<p align="center">
  <img src="https://raw.githubusercontent.com/rabiaaslam92/FIA-EDIT-DVLM/main/assets/benchmark.png" width="100%">
</p>

Metrics

- PSNR ↑
- SSIM ↑
- LPIPS ↓
- CLIP Similarity ↑

Evaluated on

- Real → Cartoon
- Real → Painting
- Real → Sketch
- Real → Real

---

#  Method

## Inherited from EEdit

- RF-Inversion
- Step Skipping
- SLoC
- Latent Composition

## Our Contributions

- Reference Forward Pass
- Reference K/V Memory
- Attention-Level K/V Injection
- Domain Prompt Augmentation
- Tail CFG
- Background Anchoring

---

#  Repository Structure

```text
FIA-EDIT-DVLM/
│
├── EEdit_colab_dvlm.ipynb      # Main notebook
├── configs/                    # Configuration files
├── ablations/                  # Ablation experiments
├── evaluation/                 # Evaluation scripts
├── assets/
│   ├── architecture.png
│   ├── qualitative_results.png
│   ├── benchmark.png
│   └── teaser.png
├── README.md
└── requirements.txt
```

---

# ⚙ Installation

```bash
git clone https://github.com/rabiaaslam92/FIA-EDIT-DVLM.git

cd FIA-EDIT-DVLM

pip install -r requirements.txt
```

---

# 📦 Models & Resources

## Base Model

- FLUX.1-dev
  - https://github.com/black-forest-labs/flux
  - https://huggingface.co/black-forest-labs/FLUX.1-dev

## Backbone

- EEdit Paper
  - https://arxiv.org/abs/2503.10270

- EEdit Project
  - https://eff-edit.github.io/

- EEdit Repository
  - https://github.com/USTC-EEIS/EEdit

## Dataset

- TF-ICON
  - https://github.com/Shilin-LU/TF-ICON

## Libraries

- HuggingFace Diffusers
  - https://github.com/huggingface/diffusers

- Transformers
  - https://github.com/huggingface/transformers

---

# ▶ Usage

Open

```
EEdit_colab_dvlm.ipynb
```

Run the notebook sequentially and adjust the configuration files for different ablation settings.

---

# 📈 Evaluation

Metrics

- PSNR
- SSIM
- LPIPS
- CLIP Similarity

Benchmarks

- Real → Cartoon
- Real → Painting
- Real → Sketch
- Real → Real

---

# 🧪 Ablation Studies

The repository contains experiments analysing

- Reference K/V injection
- Domain prompt augmentation
- Tail CFG
- Different configuration variants
- EEdit backbone settings

---

# 🙏 Acknowledgements

This project builds upon the **EEdit** framework and adapts its efficient inversion and editing strategies for reference-guided image composition. Our implementation is **not a reproduction of FIA or the original K/V injection pipeline**. Instead, we extend the EEdit backbone with attention-level reference conditioning, domain-guided prompt augmentation, and tailored modifications for FLUX-based generation.

We thank the authors of:

- EEdit
- FLUX.1
- HuggingFace Diffusers
- TF-ICON

for making their work publicly available.

---

# 📚 Citation

```bibtex
@misc{fluxkey2026,
  title={Flux & Key: Reference-Guided Image Composition with K/V Injection and Domain Guidance},
  author={Rabia Aslam, Hamna Aqeel, Syeda Attqa, Amin},
  year={2026}
}
```
