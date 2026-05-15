# Flux & Key — Reference-Guided Image Editing on FLUX.1-dev

Training-free, reference-guided image editing on FLUX.1-dev (12B rectified flow transformer).
Given a source image, a reference image containing the target object, a binary edit mask, and a
text prompt, the method edits the masked foreground while preserving the background — no
fine-tuning, no ODE inversion.

---

## Quick Start

```powershell
# Install
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers diffusers einops safetensors open_clip_torch rembg

# Place a reference image
# test_refs/000000000002_ref.jpg  ← e.g. a dog photo for cat→dog edit

# Run (recommended — fastest, cleanest result)
$env:PYTHONIOENCODING="utf-8"
python run_flux_key_test.py --place --offload --key 000000000002
```

---

## Pipeline Variants

Four Step-3 implementations with different trade-offs. Step 1 (DINO extraction) and
Step 2 (feature extraction) are shared; only Step 3 differs.

### `--place`  Direct placement + harmonization  *(recommended)*

```
rembg(ref) → PIL composite at mask bbox → noise to t=0.35 → 10-step denoising
```

- Reference is physically placed at the mask position in pixel space
- `rembg` removes background so only the object is pasted (no grass bleed)
- 10 denoising steps harmonize boundaries; text prompt guides the result
- Background latent masking keeps source chair/floor at every step (free)
- **No hooks, no extra forward passes** — ~1–2 min with `--offload`

```powershell
python run_flux_key_test.py --place --offload --key 000000000002
```

---

### `--freqref`  Frequency-domain reference blending  *(FSI-Edit inspired)*

```
rembg(ref) → PIL composite at mask bbox → noise to t=0.5 →
14-step denoising + freq_blend_V every 2 steps for t < 0.7
```

Builds on `--place` but adds frequency-domain V blending inspired by FSI-Edit:

```
V_fused = IFFT( low_freq(FFT(V_current))     ← current pose / structure
              + high_freq(FFT(V_reference)) ) ← reference breed / texture
```

Low-frequency components from the current denoising carry pose and spatial structure;
high-frequency components from the reference carry breed-specific texture and fine
appearance. Each injection pass uses the reference noised to the same timestep t
(noise-consistent — no magnitude mismatch).

```powershell
python run_flux_key_test.py --freqref --offload --key 000000000002
```

---

### `--v3`  Text-generate + late reference V injection

```
pure-noise foreground → 28-step text-guided generation →
reference V injected in all double blocks for t < 0.65 (every 2 steps)
```

Foreground starts from pure noise so FLUX is free to generate the correct pose from
text. The reference V is injected in late steps to transfer breed appearance after
pose is established. Background latent masking preserves source background.

Best for: attribute edits where the correct pose should come from text, not reference.

```powershell
python run_flux_key_test.py --v3 --offload --key 000000000002
# Override noise level (default 0.999 = full freedom):
python run_flux_key_test.py --v3 --offload --key 000000000002 --t_start 0.6
```

---

### `--v2`  FlowEdit + noise-consistent reference injection  *(research baseline)*

```
FlowEdit dual-branch: 3 forward passes/step × 28 steps = 84 passes total
```

Both source and reference branches are noised to the same timestep t at every step
(eliminating the magnitude mismatch of v1). Reference K and V are injected into the
target branch. Background K,V replaced from source at each step.

Slower (~2–3 min with offload) but theoretically the most principled approach.

```powershell
python run_flux_key_test.py --v2 --offload --key 000000000002
```

---

### (default)  SSI + KV injection  *(v1, original)*

```
Step 1 (DINO) + Step 2 (feature extraction) + Step 3 (SSI denoising with hooks)
```

Selective Stochastic Inversion noises foreground tokens at level τ; background tokens
stay clean. During denoising: background KV is frequency-locked to source features,
foreground KV is interpolated between text and reference features.

Known limitation: SSI creates a mixed-noise input (clean bg + noisy fg) that FLUX was
not trained on, causing sparkle artifacts at high τ.

```powershell
python run_flux_key_test.py --key 000000000002              # full pipeline
python run_flux_key_test.py --step3_only --key 000000000002 # reuse features.pt
python run_flux_key_test.py --step3_only --fg_inject --freq_2d --key 000000000002
```

---

## All Flags

| Flag | Applies to | Default | Description |
|------|-----------|---------|-------------|
| `--key KEY` | all | all 5 | Run only this PIE-Bench key |
| `--offload` | all | off | CPU-offload T5/CLIP/AE when idle — prevents 36 GB VRAM overflow on RTX 5090 |
| `--place` | — | off | Use direct-placement pipeline |
| `--freqref` | — | off | Use frequency-domain blending pipeline |
| `--v3` | — | off | Use text-generate + late V injection pipeline |
| `--v2` | — | off | Use FlowEdit pipeline |
| `--t_start` | `--v3` | 0.999 | Foreground noise level (0.999 = full freedom, 0.4 = preserve structure) |
| `--fg_inject` | default v1 | off | Enable foreground KV injection |
| `--freq_2d` | default v1 | off | Use 2-D spatial FFT for background lock |
| `--step3_only` | default v1 | off | Skip Steps 1–2; reuse saved `features.pt` |
| `--tau TAU` | default v1 | per-case | Override SSI noise level |

---

## Setup

### FLUX.1-dev weights

Downloaded automatically to `~/.cache/huggingface/` on first run (~30 GB).
Authenticate if required:
```bash
huggingface-cli login
```

### PIE-Bench data

```
data/pie_bench/
  mapping_file.json
  annotation_images/0_random_140/
    000000000000.jpg  ...  000000000004.jpg
```

### Reference images

```
test_refs/
  000000000002_ref.jpg   ← dog photo for cat→dog edit
  000000000001_ref.jpg   ← square cake for round→square edit
  ...
```

If a reference is missing, the source image is used as fallback (verifies pipeline
runs but produces no meaningful edit).

---

## RTX 5090 / Blackwell (SM_100)

Flash Attention and mem-efficient SDP produce NaN on Blackwell (SM_100). All pipelines
disable them automatically:

```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

**Do not re-enable Flash SDP.** It silently corrupts all K,V features with NaN.

Without `--offload`, T5 (11 GB) + CLIP (0.25 GB) + AE (0.5 GB) + FLUX (24 GB) ≈ 36 GB
exceeds the 32 GB VRAM limit, causing the GPU to throttle to ~130 W. `--offload`
keeps peak usage at ~24.5 GB and restores full clock speed.

---

## Outputs

| Output directory | Pipeline |
|-----------------|---------|
| `test_output/flux_key_place/<key>/` | `--place` |
| `test_output/flux_key_freqref/<key>/` | `--freqref` |
| `test_output/flux_key_v3/<key>/` | `--v3` |
| `test_output/flux_key_v2/<key>/` | `--v2` |
| `test_output/flux_key/<key>/` | default v1 |

Each directory contains:

| File | Description |
|------|-------------|
| `<key>_comparison.png` | Source (left) vs. edited (right), 1024×512 |
| `<key>_edited.png` | Edited image, 512×512 |
| `ref_aligned.png` | DINO-cropped reference resized to mask bbox |
| `ref_detected.jpg` | Reference with DINO detection box drawn |
| `<key>_mask.png` | RLE-decoded PIE-Bench mask |
| `features.pt` | Saved K,V features (v1 only, reuse with `--step3_only`) |

---

## File Structure

```
models/flux_key/
  step1_extract.py          # Grounding DINO: detect, crop, align reference
  step2_features.py         # Read-only hooks: extract K,V from FLUX blocks
  step3_edit.py             # v1: SSI + KV injection (original method)
  step3_flowedit.py         # v2: FlowEdit dual-branch + V injection
  step3_composite.py        # v3: text-generate + late reference V injection
  step3_place.py            # place: direct PIL composite + harmonize  ← recommended
  step3_freqref.py          # freqref: PIL composite + frequency-domain V blend
  step3_composite_v3a.py    # v3a: composite with V injection (archived)
  kv_injection.py           # KVInjectionHooks, freq_mix_1d, freq_mix_2d (v1)

run_flux_key_test.py        # 5-image test harness (all pipeline variants)
methodology.tex             # Paper: methodology section
results.tex                 # Paper: results section
```

---

## Method Comparison

| Pipeline | Pose source | Appearance source | Speed | Reference fidelity |
|----------|------------|------------------|-------|-------------------|
| `--place` | Text + scene context | Composited latent | ~1–2 min | High (direct placement) |
| `--freqref` | Text + scene context | Freq high-freq from ref | ~2 min | High |
| `--v3` | Text only | Reference V (late steps) | ~2 min | Medium |
| `--v2` | FlowEdit delta | Reference K+V (all steps) | ~3 min | Medium |
| default v1 | SSI denoising | K,V injection hooks | ~2 min | Low (NaN/sparkle at high τ) |
