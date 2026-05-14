# Flux & Key — Reference-Guided Image Editing on FLUX.1-dev

Training-free, reference-guided image editing pipeline built on FLUX.1-dev (rectified flow
transformer). Given a source image, a reference image containing the target object, a binary
edit mask, and a text prompt, the method edits the masked foreground region while preserving
the background without any model fine-tuning or ODE inversion.

---

## Method Overview

The pipeline has three stages:

```
I_source + I_reference + mask M + prompt P
        │
  ┌─────▼──────┐
  │  Step 1    │  Grounding DINO detects the target object in I_reference,
  │  Extract   │  crops it, and resizes the crop to match the spatial
  │  Reference │  extent of M in I_source → I_B_aligned
  └─────┬──────┘
        │
  ┌─────▼──────┐
  │  Step 2    │  Two clean (t=0) FLUX forward passes — one on I_source,
  │  Extract   │  one on I_B_aligned — capture K,V attention features
  │  Features  │  from all transformer blocks via read-only hooks → features.pt
  └─────┬──────┘
        │
  ┌─────▼──────┐
  │  Step 3    │  SSI: noise foreground tokens at level τ, keep background
  │  Edit      │  clean. Denoise from t=τ→0 with KV injection hooks:
  │            │    • Background tokens → frequency-locked to source K,V
  │            │    • Foreground tokens → α·K_text + (1-α)·K_ref
  └─────┬──────┘
        │
     Î_output
```

> **Note (v0.1 — current branch):** The Step 2 feature extraction at t=0 introduces a
> magnitude mismatch with noisy-timestep features during denoising. A corrected v0.2
> using FlowEdit dual-branch + per-timestep reference noising is in development.

---

## Requirements

```
Python 3.10+
PyTorch 2.6+ with CUDA 12.8 (RTX 5090 / Blackwell: Flash Attention disabled, math SDP used)
FLUX.1-dev weights (auto-downloaded via HuggingFace)
```

Install dependencies:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers diffusers einops safetensors open_clip_torch
```

---

## Setup

### 1. FLUX.1-dev weights

The pipeline uses `load_flow_model("flux-dev")` from the included `flux/` package. Weights
are downloaded automatically to `~/.cache/huggingface/` on first run (~30 GB).

Set your HuggingFace token if the model requires authentication:
```bash
huggingface-cli login
```

### 2. PIE-Bench data (optional, for evaluation)

```
data/
  pie_bench/
    mapping_file.json
    annotation_images/
      0_random_140/
        000000000000.jpg
        000000000001.jpg
        ...
```

### 3. Reference images

Place reference images in `test_refs/`:
```
test_refs/
  000000000002_ref.jpg   ← dog image for cat→dog edit
  000000000001_ref.jpg   ← square cake for round→square edit
  ...
```

If no reference image is found for a key, the source image is used as fallback
(validates pipeline flow only — no meaningful object change will occur).

---

## Running the Pipeline

### Single image — full pipeline (Steps 1 + 2 + 3)

```powershell
$env:PYTHONIOENCODING="utf-8"
python run_flux_key_test.py --key 000000000002
```

### Single image — Step 3 only (reuse saved features.pt)

```powershell
python run_flux_key_test.py --step3_only --key 000000000002
```

### Ablations

```powershell
# Background lock only (no reference injection) — validates background preservation
python run_flux_key_test.py --step3_only --key 000000000002

# Full method with foreground reference injection
python run_flux_key_test.py --step3_only --fg_inject --key 000000000002

# Use 2-D spatial FFT for background lock (experimental; compare with default 1-D)
python run_flux_key_test.py --step3_only --fg_inject --freq_2d --key 000000000002

# Override tau for all test cases (attribute edits: 0.3–0.4; object replacement: 0.6–0.75)
python run_flux_key_test.py --step3_only --fg_inject --key 000000000002 --tau 0.4

# Run all 5 test cases
python run_flux_key_test.py --fg_inject --freq_2d
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--key KEY` | all 5 | Run only this PIE-Bench sample key |
| `--tau TAU` | per-case | Override noise level (0.0–1.0); higher = more aggressive |
| `--fg_inject` | off | Enable foreground reference K,V injection |
| `--freq_2d` | off | Use 2-D spatial FFT for background lock (default: 1-D) |
| `--step3_only` | off | Skip Steps 1–2; reuse saved `features.pt` |

---

## RTX 5090 / Blackwell (SM_100) notes

Flash Attention produces NaN on Blackwell architecture. The pipeline disables it
automatically in both feature extraction (Step 2) and denoising (Step 3):

```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

Do NOT re-enable Flash SDP; doing so will silently corrupt features.pt with NaN values
that propagate into every subsequent denoising step.

---

## Outputs

Each run saves to `test_output/flux_key/<key>/`:

| File | Description |
|------|-------------|
| `<key>_edited.png` | Edited image (512×512) |
| `<key>_comparison.png` | Side-by-side: source (left) vs. edited (right) |
| `features.pt` | Saved Step 2 features (reuse with `--step3_only`) |
| `ref_aligned.png` | DINO-cropped, resized reference object |
| `ref_detected.jpg` | Reference image with DINO detection box drawn |
| `<key>_mask.png` | RLE-decoded PIE-Bench mask (visual check) |

---

## Key hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `tau` | 0.4–0.75 | Noise level for foreground tokens. Higher = more object change, less background fidelity |
| `alpha_max` | 0.5 | Maximum text weight in α·K_text + (1-α)·K_ref. Lower = more reference influence |
| `alpha_min` | 0.1 | Minimum text weight (at late blocks, late timesteps) |
| `gamma` | 0.5 | Frequency split for background lock. Lower = more source high-freq retained |
| `guidance` | 3.5 | FLUX guidance scale |
| `num_steps` | 28 | Total denoising steps; pipeline starts from step index where t ≤ tau |

---

## Architecture: KV Injection Hooks

All editing logic is implemented as **PyTorch forward hooks** that intercept the QKV
projection outputs inside FLUX transformer blocks. The original FLUX model files are
not modified.

```
FLUX DoubleStreamBlock.img_attn.qkv  →  _make_double_img_hook(block_idx)
FLUX SingleStreamBlock.linear1       →  _make_single_hook(block_idx)
```

At each denoising step, the hooks:

1. **Background tokens** (mask==0): replace K,V with `freq_mix(K_src, K_tgt, γ_t)` where
   γ_t = γ + (1−γ)·t (time-aware: no lock at t≈1, full lock at t=0)
2. **Foreground tokens** (mask==1): interpolate `α_t^(ℓ)·K_text + (1−α_t^(ℓ))·K_ref`
   with layer-wise + time-aware α schedule

Hooks are attached before denoising and removed after — no persistent model modification.

---

## Known Limitations (v0.1)

1. **Mixed-noise inconsistency**: SSI noises only foreground tokens; FLUX expects all
   tokens at the same noise level. This causes sparkle artifacts at high τ. Fix planned
   in v0.2 using FlowEdit dual-branch paradigm.

2. **Feature timestamp mismatch**: K,V features are extracted at t=0 (clean pass) but
   used during denoising at t>0. This causes magnitude mismatch and occasional NaN
   (clamped to 0). Fix: noise reference to current timestep t before each extraction.

3. **V injection in MM-DiT**: Per QK-Edit (ICCV 2025), replacing V in FLUX's joint
   attention can suppress editability. Future version will use Q,K manipulation only,
   or restrict injection to content-similarity layers (FreeFlux layer analysis).

---

## File Structure

```
models/flux_key/
  __init__.py
  step1_extract.py     # Grounding DINO reference extraction
  step2_features.py    # K,V feature extraction (read-only hooks)
  step3_edit.py        # SSI + KV injection denoising
  kv_injection.py      # KVInjectionHooks, freq_mix_1d, freq_mix_2d

run_flux_key_test.py   # 5-image end-to-end test harness
methodology.tex        # Paper methodology section (LaTeX)
results.tex            # Paper results section (LaTeX)
```
