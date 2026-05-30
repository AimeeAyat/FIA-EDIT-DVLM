# EEdit Composition Pipeline: Original Design and Our Extensions

---

## Part I — Original EEdit Pipeline

---

### 1. Overview

EEdit is an **inversion-based image editing framework** built on top of FLUX.1-dev, a rectified-flow diffusion transformer. Its key contribution is reducing the severe computational overhead of inversion-based editing through two orthogonal techniques: **temporal redundancy elimination** via inversion step skipping and RF-Inversion trajectory guidance, and **spatial redundancy elimination** via spatial locality caching with token importance scoring. Together these achieve a reported average of 2.46× speedup over the uncached inversion baseline with negligible quality degradation.

The composition task specifically places a foreground reference object (segmented from a source image) into a target background scene while preserving the background's structure and blending the object naturally into the scene. The four benchmark domains are:

- **RC (Real-Cartoon)**: photorealistic foreground composited into a cartoon/vector-art background
- **RP (Real-Painting)**: photorealistic foreground into a painted background
- **RS (Real-Sketch)**: photorealistic foreground into a hand-drawn sketch background
- **RR (Real-Real)**: photorealistic foreground into a photorealistic background

---

### 2. FLUX.1-dev Architecture

FLUX.1-dev is a **Multimodal Diffusion Transformer (MMDiT)** trained on the rectified flow objective. It processes two token streams jointly:

- **Image tokens** $x \in \mathbb{R}^{B \times S_\text{img} \times D}$: VAE-encoded image latents packed into 2×2 spatial patches, where $S_\text{img} = (H/16) \times (W/16)$. For 512×512 images: $S_\text{img} = 32 \times 32 = 1{,}024$ tokens.
- **Text tokens** $c \in \mathbb{R}^{B \times S_\text{txt} \times D}$: T5-XXL text encoder outputs, $S_\text{txt} \leq 512$ tokens.

The transformer consists of:
- **19 joint (double-stream) blocks**: process $x$ and $c$ in separate streams with cross-attention between them. These blocks handle semantic content, appearance, and spatial layout.
- **38 single-stream blocks**: process only $x$ after text features have been merged in. These handle fine-grained spatial structure and texture.

Each joint block computes multi-head self-attention over the concatenated sequence $[c;\, x]$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

where $Q, K, V \in \mathbb{R}^{B \times H \times (S_\text{txt}+S_\text{img}) \times d_k}$, $H$ is the number of heads, and $d_k = D/H$. Rotary positional embeddings (RoPE) are applied to $Q$ and $K$ before the dot product.

**VAE encoding**: images are encoded via a 16× downsampling VAE. A 512×512 RGB image maps to a $4 \times 32 \times 32$ latent, then packed into 1,024 tokens of dimension 64 (4 channels × 2×2 patch). The pipeline applies shift and scale factors:

$$z = (z_\text{raw} - \mu_\text{shift}) \times s_\text{scale}$$

---

### 3. Rectified Flow Formulation

FLUX uses **Rectified Flow (RF)**, a generative framework defining a straight-line interpolation between data $x_0$ and noise $x_1 \sim \mathcal{N}(0, I)$:

$$x_t = (1 - t)\cdot x_0 + t \cdot x_1, \quad t \in [0, 1]$$

The model $f_\theta$ is trained to predict the velocity field:

$$v^* = x_1 - x_0$$

The denoising ODE is:

$$\frac{dx_t}{dt} = f_\theta(x_t,\, t,\, c)$$

To generate an image, one solves this ODE backward from $t=1$ (pure noise) to $t=0$ (clean image) using $N$ Euler steps:

$$x_{t_{i-1}} = x_{t_i} + (t_{i-1} - t_i)\cdot f_\theta(x_{t_i},\, t_i,\, c)$$

---

### 4. RF-Inversion (Controlled Forward ODE)

Standard diffusion editing requires inverting the image back to the noise space before re-denoising with a new prompt. For rectified flow, inversion means integrating the ODE *forward* from $t=0$ (clean image) to $t=1$ (noise).

**RF-Inversion**, implemented in EEdit's `controlled_forward_ode`, introduces a **guidance signal** that corrects the inversion trajectory using the analytical conditional velocity. The analytical conditional velocity at step $i$ — what the velocity *should* be to reach the fixed noise sample $y_1$ from the current state $Y_{t_i}$ in the remaining time — is:

$$u_{t_i}^\text{cond} = \frac{y_1 - Y_{t_i}}{1 - t_i}$$

where $y_1 \sim \mathcal{N}(0, I)$ is drawn once at the start of inversion. The unconditioned model prediction is:

$$u_{t_i} = f_\theta(Y_{t_i},\; t_i,\; \varnothing)$$

These are blended via the guidance parameter $\gamma$:

$$\hat{u}_{t_i} = u_{t_i} + \gamma \cdot \left(u_{t_i}^\text{cond} - u_{t_i}\right)$$

The latent is then advanced:

$$Y_{t_{i+1}} = Y_{t_i} + \hat{u}_{t_i} \cdot \Delta t, \quad \Delta t = \frac{1}{N}$$

**Effect of $\gamma$**:
- $\gamma = 0$: pure unconditional inversion — trajectory is unconstrained, poor reconstruction
- $\gamma = 1$: full guidance toward the analytical ODE solution — tight reconstruction
- Intermediate $\gamma$: balances faithfulness to the analytical trajectory against flexibility for editing

At the end of inversion, $Y_1$ (the inverted noise latent) is used as the starting point for the denoising phase.

**Code reference** (`MyCodes/MyFluxCompositionPipeline.py:211–231`):
```python
for i in range(N-1):
    t_i = i / N
    if i % skip_T == 0 or i == N-2:          # step skipping
        u_t_i = transformer(Y_t, t_i, null_prompt)
    u_t_i_cond = (y_1 - Y_t) / (1 - t_i)    # analytical velocity
    u_hat_t_i = u_t_i + gamma * (u_t_i_cond - u_t_i)
    Y_t = Y_t + u_hat_t_i * dt
```

---

### 5. Inversion Step Skipping

Running the transformer at every inversion step is expensive. EEdit introduces **inversion step skipping** controlled by `inv_skip` (`skip_T` in code). The transformer is only called at every `inv_skip`-th step; between calls the previously computed $u_{t_i}$ is reused:

$$u_{t_i} = \begin{cases} f_\theta(Y_{t_i},\, t_i,\, \varnothing) & \text{if } i \bmod \texttt{skip\_T} = 0 \text{ or } i = N-2 \\ u_{t_{i-\text{prev}}} & \text{otherwise (reuse)} \end{cases}$$

The $i = N-2$ condition ensures the final inversion step is always freshly computed to prevent divergence at the noise boundary.

**Effect**: with `inv_skip=2` (default) and $N=28$, only 14 transformer calls are made during inversion instead of 27. The analytical correction $u_{t_i}^\text{cond}$ is computed at every step regardless (it requires no network call), so trajectory guidance remains active throughout.

---

### 6. RF-Inversion Guided Denoising

After inversion, denoising proceeds from the inverted latent $Y_1$. At each denoising step $i$, the model produces a noise prediction $\varepsilon_\text{pred}$, and the **inversion trajectory is injected back** to prevent the output from diverging too far from the original image:

$$v_t^\text{inv} = \frac{x_0^\text{composite} - z_{t_i}}{1 - t_i}$$

where $x_0^\text{composite}$ is the composite image latent (background + pasted foreground). The guided velocity is:

$$\hat{v}_t = -\varepsilon_\text{pred} + \eta_t \cdot \left(v_t^\text{inv} + \varepsilon_\text{pred}\right)$$

where the time-gated injection strength is:

$$\eta_t = \begin{cases} \eta & \text{if } \texttt{start\_timestep} \leq i < \texttt{stop\_timestep} \\ 0 & \text{otherwise} \end{cases}$$

**Interpretation**:
- $\eta_t = 0$: $\hat{v}_t = -\varepsilon_\text{pred}$ — standard denoising, maximum FLUX creative freedom
- $\eta_t = 1$: $\hat{v}_t = v_t^\text{inv}$ — model forced to follow inversion trajectory exactly, maximum fidelity to original

The denoising step then advances the latent:

$$z_{t_{i-1}} = \text{Scheduler.step}(-\hat{v}_t,\; t_i,\; z_{t_i})$$

After each denoising step, **background anchoring** hard-pins tokens outside the bounding box:

$$z_{t_{i-1}} \leftarrow (1 - M) \cdot \tilde{x}_0^\text{bg} + M \cdot z_{t_{i-1}}$$

where $M \in \{0,1\}^{S_\text{img}}$ is the bounding-box mask (1 inside edit region, 0 outside), and $\tilde{x}_0^\text{bg}$ is the background latent noised to the appropriate $\sigma_i$ level. This prevents background drift at every step regardless of other settings.

**Code reference** (`MyCodes/MyFluxCompositionPipeline.py:1150–1168`):
```python
v_t_cond = (y_0 - gen_latents) / (1 - t_i)
eta_t = eta if start_timestep <= i < stop_timestep else 0.0
v_hat_t = -noise_pred + eta_t * (v_t_cond + noise_pred)
gen_latents = scheduler.step(-v_hat_t, t, gen_latents)
gen_latents = (1 - bbox_mask) * bg_latents_at_sigma_i + bbox_mask * gen_latents
```

---

### 7. Spatial Locality Caching

The central speed contribution of EEdit is the observation that during editing, **only the tokens within and immediately around the edit region need to be fully recomputed**; tokens in unchanged background regions produce nearly identical attention features across steps and can be served from cache.

#### 7.1 Token Importance Scoring

The edit region is defined by bounding box $(x_1, y_1, x_2, y_2)$. In latent space (scaled by factor 16), this maps to token index set $\mathcal{E}_0$. A cascade of $K$ neighbour-expansion levels is computed:

$$\mathcal{E}_k = \mathcal{E}_{k-1} \cup \left\{(i,j) \;\middle|\; \exists\,(i',j') \in \mathcal{E}_{k-1} : \|(i,j)-(i',j')\|_\infty \leq 1\right\}, \quad k = 1,\ldots,K-1$$

Each token $n$ is assigned an importance score:

$$s_n = 1 + \texttt{edit\_base} \times \sum_{k=0}^{K-1} \texttt{bonus\_ratio}^{1+k} \cdot \mathbf{1}[n \in \mathcal{E}_k \setminus \mathcal{E}_{k-1}]$$

With `edit_base=2` and `bonus_ratio=0.8`:
- Core edit tokens $\mathcal{E}_0$: score $= 1 + 2 \times 0.8^1 = 2.6$
- First ring $\mathcal{E}_1 \setminus \mathcal{E}_0$: score $= 1 + 2 \times 0.8^2 = 2.28$
- Second ring: score $= 1 + 2 \times 0.8^3 = 2.024$
- Background: score $= 1.0$

#### 7.2 Cache Refresh Mechanism

At each denoising step, only the top `fresh_ratio` fraction of tokens (by descending importance score) are recomputed. The remainder are served from the previous step's cache.

A **hard refresh** occurs every `fresh_threshold` steps, unconditionally recomputing all tokens to prevent stale cache accumulation.

**Soft refresh** blends newly computed attention with cached value:

$$a_n^\text{new} = w \cdot a_n^\text{fresh} + (1-w) \cdot a_n^\text{cached}, \quad w = \texttt{soft\_fresh\_weight}$$

#### 7.3 Token Indexing Preprocessing (TIP) — `ours_predefine`

The `ours_predefine` cache type pre-computes the exact set of token indices to refresh at each denoising step *before* generation begins via `predefine_cache_fresh_indices`. This eliminates per-step sorting overhead, yielding additional speedup over the dynamic `ours_cache` variant.

#### 7.4 Tailing Step

`tailing_step=1` (fixed in paper): the number of trailing denoising steps that perform a final full-cache refresh to clean up residual caching artifacts before decoding.

---

### 8. Complete Composition Pipeline Flow

```
Input:  background image, foreground image, foreground mask,
        prompt, bounding box (x1, y1, x2, y2)

1. ENCODE
   bg_latent    = VAE.encode(background)              [B, 4, 32, 32]
   ref_latent   = VAE.encode(fg × mask)
   ref_resized  = interpolate(ref_masked, (y2−y1, x2−x1))

2. COMPOSITE
   composite = bg_latent.clone()
   composite[:, :, y1:y2, x1:x2] = ref_resized       # paste fg into bg

3. CACHE INIT
   edit_idx  = edit_region_parser(x1,y1,x2,y2, cascade_num=5)
   score     = convert_to_cache_index(edit_idx)
   cache_dic, current = cache_init(model_kwargs, N)
   predefine_cache_fresh_indices(cache_dic, current)  # TIP

4. RF-INVERSION  [controlled_forward_ode]
   Y = composite_latent;  y_1 ~ N(0,I)
   for i in 0..N−1:
       if i % inv_skip == 0:  u_t = transformer(Y, t, null_prompt)
       u_cond = (y_1 − Y) / (1 − t)
       Y = Y + [u_t + γ(u_cond − u_t)] × Δt

5. DENOISING LOOP  [28 steps, with caching]
   gen = Y_1  (inverted noise)
   for i in 0..27:
       noise_pred = transformer(gen, t_i, prompt)     # cached
       v_hat = −noise_pred + η_t × (v_inv + noise_pred)
       gen   = scheduler.step(−v_hat, t, gen)
       gen   = (1−M)×bg_at_σ_i + M×gen               # background anchor

6. DECODE
   image = VAE.decode(gen)
```

---

### 9. Original Configuration Parameters

All parameters that are unchanged between paper and our implementation are listed here with their mathematical roles.

| Parameter | Paper Value | Mathematical Role |
|-----------|-------------|-------------------|
| `num_inference_steps` | 28 | Total denoising steps $N$ |
| `inv_skip` | 2 | Inversion skip interval; transformer called at every `inv_skip`-th step |
| `start_timestep` | 0 | Lower bound of η injection window; fixed |
| `blend_ratio` | 0 | Foreground–background alpha blend at composite stage; deprecated |
| `use_rf_inversion` | true | Enable/disable RF-Inversion trajectory injection |
| `cascade_num` | 5 | Number of neighbour-expansion levels $K$ for token importance scoring |
| `fresh_ratio` | 0.1 | Fraction of tokens recomputed per cached denoising step |
| `fresh_threshold` | 3 | Every this many steps, force full cache refresh |
| `soft_fresh_weight` | 0.25 | Blend weight $w$ for soft cache refresh |
| `tailing_step` | 1 | Number of trailing full-refresh steps |
| `use_cache` | true | Enable/disable spatial locality caching |
| `cache_type` | `ours_predefine` | TIP (pre-indexed) vs `ours_cache` (dynamic per-step sort) |
| `edit_base` | 2 (code fixed) | Importance score base multiplier |

The paper uses **domain-specific** values for the core editing parameters (`eta`, `gamma`, `stop_timestep`), documented accurately in Section 14 below.

---

## Part II — Our Modified Pipeline

---

### 10. Overview of Modifications

Three additions extend the base EEdit composition pipeline. All changes are instance-level — installed before each image and removed after, with no permanent modifications to the transformer class or scheduler:

| Module | Addition | Scope |
|--------|----------|-------|
| `dvlm/prompt_utils.py` | Domain-aware prompt augmentation | Text conditioning |
| `dvlm/ref_inject.py` | Reference K,V attention injection | Attention computation |
| `dvlm/pipeline_patches.py` | Tail classifier-free guidance | Last N denoising steps |

---

### 11. Domain-Aware Prompt Augmentation

#### 11.1 Motivation

The four composition domains differ fundamentally in their visual priors. FLUX.1-dev's text conditioning is strong enough that appending style-specific language meaningfully shifts the generation prior toward the target domain — without this, the model may default to a generic photorealistic composite that conflicts with cartoon, painting, or sketch backgrounds.

#### 11.2 Implementation

The augmented prompt is:

$$p_\text{aug} = p_\text{original} \;\|\; s_\text{domain}$$

Domain-specific positive suffixes:

| Domain | Suffix |
|--------|--------|
| **RC** | `"cartoon style, bold thick black outlines, flat vector colors, exaggerated expressions. Original face, identity, and proportions preserved."` |
| **RP** | `"realistic painting, featuring visible textures, artistic lighting, and authentic paint strokes. Original face, identity, and proportions preserved."` |
| **RS** | `"Sketch style. hand-drawn, featuring realistic line work, manual shading, and authentic paper texture. Original face, identity, and proportions preserved."` |
| **RR** | `"Photorealistic style. Seamless integration into the scene. Matching lighting, shadows, textures, perspective, and scale. Original face, identity, and proportions preserved."` |

Domain-specific **negative prompts** are also defined and used by the tail-CFG module. For example, the RC negative prompt penalizes: `"photorealistic patches, inconsistent line thickness, pasted appearance, hard edges, cut-out look"` — exactly the artifacts that arise when a photo-realistic object is naively composited into a cartoon scene.

#### 11.3 Latency Impact

Prompt augmentation is a Python string concatenation. One additional T5 encoding call is made per image for the negative prompt (used by tail-CFG). **Marginal latency: ≈ 5–10 ms**.

---

### 12. Reference K,V Injection

#### 12.1 Motivation

RF-Inversion with $\eta=0.75$ provides strong scene grounding but operates on the composite image (background + pasted foreground). In cross-domain settings (RC, RP, RS), FLUX's domain prior aggressively re-renders the foreground object's color and texture to match the background style (a cartoon panda instead of a real-looking panda). A direct mechanism is needed to inject the reference object's appearance into attention without interfering with scene integration.

#### 12.2 Pre-Extraction of Reference K,V

A single forward pass through the first `ref_inject_blocks` joint blocks is run on the reference image to capture intermediate hidden states, which are immediately projected into key-value pairs. This **one-time extraction** amortizes projection cost across all denoising steps.

**Step 1 — VAE encode:**
$$z_\text{ref} = \text{VAE.encode}(\text{resize}(I_\text{ref},\, 512\times512))$$
$$z_\text{ref} \leftarrow (z_\text{ref} - \mu_\text{shift}) \times s_\text{scale}$$

**Step 2 — Pack into FLUX token format:**
$$h_\text{ref} = \text{pack}(z_\text{ref}) \in \mathbb{R}^{1 \times 1024 \times 64}$$

Image position IDs $P_\text{ref} \in \mathbb{R}^{1024 \times 3}$ are constructed with $P[i,0]=0$ (batch), $P[i,1]=\lfloor i/32 \rfloor$ (row), $P[i,2] = i \bmod 32$ (column).

**Step 3 — Hook-based capture:**

Forward hooks are registered on `to_k` of blocks $0, 1, \ldots, \texttt{ref\_inject\_blocks}-1$. A single uncached forward pass is run. The hooks record hidden states $h_i^\text{raw} \in \mathbb{R}^{1 \times S_\text{ref} \times D}$ at the `to_k` input for each block $i$.

**Step 4 — Pre-projection (one-time cost):**

For each captured block $i$:
$$K_i^\text{ref} = \text{norm}_k\!\left(\text{to\_k}(h_i^\text{raw})\right) \in \mathbb{R}^{1 \times H \times S_\text{ref} \times d_k}$$
$$V_i^\text{ref} = \text{to\_v}(h_i^\text{raw}) \in \mathbb{R}^{1 \times H \times S_\text{ref} \times d_k}$$

These are stored in `{block_idx: (K_ref, V_ref)}` and moved to GPU per step as needed.

**Why pre-projection matters**: if $K^\text{ref}, V^\text{ref}$ were computed inside the processor (at every denoising step), the cost would be $2 \times \texttt{ref\_inject\_blocks} \times \texttt{ref\_inject\_steps}$ additional linear projections on the hot path. Pre-projecting once reduces this to a single batch at extraction time plus $\texttt{ref\_inject\_blocks} \times \texttt{ref\_inject\_steps}$ cheap `torch.cat` calls.

#### 12.3 Per-Step Injection via RefInjectAttnProcessor

For denoising step $t \in [0, \texttt{ref\_inject\_steps})$ and block $i \in [0, \texttt{ref\_inject\_blocks})$, the standard attention processor is replaced. The processor computes:

**Image stream:**
$$Q_\text{img} = \text{norm}_q(\text{to\_q}(h_\text{img})) \;\in\; \mathbb{R}^{B \times H \times S_\text{img} \times d_k}$$
$$K_\text{img} = \text{norm}_k(\text{to\_k}(h_\text{img})), \quad V_\text{img} = \text{to\_v}(h_\text{img})$$

**Text stream:**
$$Q_\text{txt} = \text{norm}_{q,\text{add}}(\text{add\_q\_proj}(h_\text{txt})) \;\in\; \mathbb{R}^{B \times H \times S_\text{txt} \times d_k}$$
$$K_\text{txt} = \text{norm}_{k,\text{add}}(\text{add\_k\_proj}(h_\text{txt})), \quad V_\text{txt} = \text{add\_v\_proj}(h_\text{txt})$$

**Concatenate text and image, apply RoPE:**
$$Q = \text{cat}([Q_\text{txt},\, Q_\text{img}],\,\text{dim}=2) \;\in\; \mathbb{R}^{B \times H \times (S_\text{txt}+S_\text{img}) \times d_k}$$
$$K = \text{RoPE}(\text{cat}([K_\text{txt},\, K_\text{img}],\,\text{dim}=2))$$
$$V = \text{cat}([V_\text{txt},\, V_\text{img}],\,\text{dim}=2)$$

**Append pre-projected reference K,V after RoPE** (reference tokens use their extraction-time position encoding):
$$K_\text{ext} = \text{cat}([K,\, K_i^\text{ref}],\,\text{dim}=2) \;\in\; \mathbb{R}^{B \times H \times (S_\text{txt}+S_\text{img}+S_\text{ref}) \times d_k}$$
$$V_\text{ext} = \text{cat}([V,\, V_i^\text{ref}],\,\text{dim}=2)$$

**Scaled dot-product attention:**
$$A = \text{SDPA}(Q,\, K_\text{ext},\, V_\text{ext}) = \text{softmax}\!\left(\frac{Q K_\text{ext}^\top}{\sqrt{d_k}}\right) V_\text{ext}$$
$$A \in \mathbb{R}^{B \times H \times (S_\text{txt}+S_\text{img}) \times d_k}$$

Note: $Q$ has length $S_\text{txt}+S_\text{img}$ so the output has the same length. Reference tokens serve as additional keys and values only — they contribute to what image tokens attend to but do not produce output tokens.

**Split and project outputs:**
$$\text{out}_\text{img} = \text{to\_out}[1](\text{to\_out}[0](A[:,\,:,\,S_\text{txt}:,\,:]))$$
$$\text{out}_\text{txt} = \text{to\_add\_out}(A[:,\,:,\,:S_\text{txt},\,:])$$

**Effect**: every image token can attend to both its standard context and the full reference image representation. This biases the attention output of the edit region toward the reference object's appearance, anchoring color and texture identity throughout the structural phase of denoising.

---

### 13. Tail Classifier-Free Guidance

#### 13.1 Motivation

Standard CFG doubles forward-pass cost at every step. EEdit's base pipeline supports CFG via `do_cfg=True` but does not use it by default for composition. We introduce **tail CFG** — guidance applied only in the last `cfg_tail_steps` denoising steps — targeting the quality–cost tradeoff. Final denoising steps determine fine-grained style, texture sharpening, and edge definition: the highest-leverage window for guidance at lowest total cost.

#### 13.2 Standard CFG Equation

$$\hat{v}_t = v_\theta(z_t,\, t,\, c_\text{neg}) + s \cdot \left[v_\theta(z_t,\, t,\, c_\text{pos}) - v_\theta(z_t,\, t,\, c_\text{neg})\right]$$

where $s = \texttt{cfg\_scale}$, $c_\text{neg}$ is the domain-specific negative prompt embedding, $c_\text{pos}$ is the augmented positive prompt embedding.

#### 13.3 Tail Window Gate

Let $N = \texttt{num\_inference\_steps}$ and $T_\text{start} = N - \texttt{cfg\_tail\_steps}$:

$$\hat{v}_t = \begin{cases} v_\theta(z_t,\, t,\, c_\text{pos}) & i < T_\text{start} \\ v_\theta(z_t,\, t,\, c_\text{neg}) + s\cdot[v_\theta(z_t,\, t,\, c_\text{pos}) - v_\theta(z_t,\, t,\, c_\text{neg})] & i \geq T_\text{start} \end{cases}$$

#### 13.4 Inversion-Phase Detection

The patch wraps `pipe.transformer.forward` and must not activate during inversion (`controlled_forward_ode`), which also calls the transformer. The detection mechanism exploits: `current['step']` is incremented during inversion (0, 1, …, N−2), then the denoising loop **resets it to 0** at $i=0$. The patch tracks:

- `_prev_step`: last observed value of `current['step']`
- `_in_denoising`: flag set to `True` when `_prev_step > 0` and `step == 0` is observed

This cleanly separates inversion from denoising with no pipeline modification.

#### 13.5 Sequential Passes

`FluxTransformer2DModel_PREDEFINE` contains a gate tensor of shape `[batch, D]`. For batch=1: PyTorch broadcasts as `[1, 1, D]` against `[1, seq, D]` — correct. For batch=2 (standard batched CFG trick): shape `[1, 2, D]` vs `[2, seq, D]` — dimension-1 mismatch (`2 ≠ seq_len`), raising `RuntimeError`. We therefore use two **sequential** forward passes (neg then pos) rather than a batched pass. The cost is one extra full uncached forward per tail step.

---

### 14. Configuration Changes — Verified Against Paper

#### 14.1 Accurate Paper vs Ours Comparison

The paper uses **domain-specific** values for `eta`, `gamma`, and `stop_timestep`. These are read directly from `configs/composition/{RC,RP,RS,RR}_config.json`.

| Parameter | Paper RC | Ours RC | Paper RP | Ours RP | Paper RS | Ours RS | Paper RR | Ours RR |
|-----------|----------|---------|----------|---------|----------|---------|----------|---------|
| `eta` | 0.50 | **0.75** | 0.75 | 0.75 ✓ | 0.60 | **0.75** | 1.00 | **0.75** |
| `gamma` | 0.50 | **0.75** | 0.75 | 0.75 ✓ | 0.60 | **0.75** | 1.00 | **0.75** |
| `stop_timestep` | 18 | **13** | 15 | **12** | 10 | **22** | 14 | 14 ✓ |
| `ref_inject_blocks` | — | 16 | — | 16 | — | 12 | — | 6 |
| `ref_inject_steps` | — | 18 | — | 18 | — | 14 | — | 10 |
| `cfg_tail_steps` | — | 5 | — | 7 | — | 5 | — | 4 |
| `cfg_scale` | — | 3.5 | — | 5.0 | — | 3.0 | — | 2.5 |
| `bbox_margin` | — | 0.10 | — | 0.10 | — | — | — | — |

All other parameters (`num_inference_steps`, `cascade_num`, `fresh_ratio`, `fresh_threshold`, `soft_fresh_weight`, `tailing_step`, `inv_skip`, `cache_type`, `blend_ratio`, `start_timestep`, `use_rf_inversion`, `use_cache`) are **identical to paper in all domains**.

#### 14.2 Domain-by-Domain Rationale

**RC (Real-Cartoon)**
- η/γ: **0.50 → 0.75** (+50%). With ref_inject providing foreground identity, a stronger inversion signal is needed to prevent the background from drifting under the combined pull of ref K,V + strong CFG (scale=3.5).
- stop_timestep: **18 → 13** (−5 steps). Shortening the injection window frees 15 denoising steps (steps 13–28) for FLUX to cartoon-stylize the subject. The paper's 18-step window was suitable without ref_inject but over-constrains generation when appearance is already anchored by K,V injection.
- bbox_margin=0.10: expand paste region 10% per side so FLUX-generated subject has edge blending room.

**RP (Real-Painting)**
- η/γ: **unchanged at 0.75**. The paper already chose 0.75 for painting — matching our approach.
- stop_timestep: **15 → 12** (−3 steps). Modest shortening consistent with RC reasoning.
- cfg_scale=5.0: highest across all domains, needed because the painting style prior is visually distinctive and requires strong prompt pressure to impose on the composited subject.

**RS (Real-Sketch)**
- η/γ: **0.60 → 0.75** (+25%). Raised to better preserve the structural simplicity of sketch backgrounds, which dissolve easily under lower-η denoising.
- stop_timestep: **10 → 22** (most dramatic change, +12 steps). The paper used only 10 injection steps for RS. We inject for 22 of 28 steps (79%). Sketch scenes have minimal texture, and without prolonged inversion injection the line-art background structure breaks down completely. The 22-step window keeps the sketch geometry grounded throughout denoising.
- No bbox_margin (sketch subjects have cleaner masking).

**RR (Real-Real)**
- η/γ: **1.00 → 0.75** (−25%). Paper used full reconstruction (η=1.0 means denoising always follows the inversion trajectory exactly — zero FLUX creative freedom). We reduced to 0.75 to allow natural shadow/lighting blending, compensated by adding ref_inject (which the paper did not use for RR).
- stop_timestep: **unchanged at 14**.
- cfg_scale=2.5: lowest, as same-domain composition needs minimal style push.

---

### 15. Latency Analysis

#### 15.1 Baseline (Original EEdit)

**Measured average: 4.6 s per image** (28 steps, `ours_predefine`, `inv_skip=2`).

Let $\tau_f$ = cost of one full uncached transformer forward pass.

| Phase | Passes | Cached? | Effective cost |
|-------|--------|---------|----------------|
| RF-Inversion | $\lceil 28/2 \rceil = 14$ | No | $14\,\tau_f$ |
| Denoising — cached steps (~25) | 25 | Yes (10% tokens) | $25 \times 0.1\,\tau_f = 2.5\,\tau_f$ |
| Denoising — hard-refresh steps (every 3rd → ~3) | 3 | No | $3\,\tau_f$ |
| **Total** | | | $\approx 19.5\,\tau_f$ |

$$\tau_f \approx \frac{4.6}{19.5} \approx 0.236 \text{ s/pass}$$

#### 15.2 Additional Costs of Our Pipeline

**Addition 1 — ref_inject extraction** (once per image, uncached, all joint blocks):
$$\Delta t_\text{extract} = 1 \times \tau_f \approx 0.24 \text{ s}$$

**Addition 2 — RefInjectAttnProcessor cache bypass**:

During the first `ref_inject_steps` denoising steps, `RefInjectAttnProcessor` performs full manual Q,K,V computation for each of the `ref_inject_blocks` joint blocks, bypassing the EEdit cache. Additionally, the attention sequence length doubles from $S_\text{txt}+S_\text{img}$ to $S_\text{txt}+S_\text{img}+S_\text{ref}$:

$$C_\text{attn}^\text{ours} \propto (S_\text{txt}+S_\text{img}+S_\text{ref}) \times (S_\text{txt}+S_\text{img}) = (512+2048) \times (512+1024) \approx 3.84 \times 10^6$$
$$C_\text{attn}^\text{orig} \propto (S_\text{txt}+S_\text{img})^2 = (512+1024)^2 \approx 2.36 \times 10^6$$

Ratio: $\approx 1.63\times$ more expensive per active block per step.

For RC (16 blocks, 18 steps, `fresh_ratio`=0.1, joint blocks ≈ 33% of $\tau_f$, 19 joint blocks):
$$\Delta t_\text{inject} \approx 18 \times 16 \times \frac{0.33\,\tau_f}{19} \times (1 - 0.1) \times 1.63 \approx 1.3\,\tau_f \approx 0.31 \text{ s}$$

**Addition 3 — cfg_tail sequential passes** (last `cfg_tail_steps` steps, 1 extra uncached pass each):
$$\Delta t_\text{CFG} = \texttt{cfg\_tail\_steps} \times \tau_f$$

| Domain | cfg_tail_steps | $\Delta t_\text{CFG}$ |
|--------|---------------|----------------------|
| RC | 5 | $5 \times 0.236 = 1.18$ s |
| RP | 7 | $7 \times 0.236 = 1.65$ s |
| RS | 5 | $5 \times 0.236 = 1.18$ s |
| RR | 4 | $4 \times 0.236 = 0.94$ s |
| **Average** | 5.25 | **1.24 s** |

**Addition 4 — Extra prompt encodings** (neg + pos T5 calls per image):
$$\Delta t_\text{encode} \approx 0.10 \text{ s}$$

#### 15.3 Predicted vs Measured

| Component | Cost (s) |
|-----------|----------|
| ref_inject extraction | +0.24 |
| Cache bypass + extended attention (inject processor) | +0.31 |
| cfg_tail average across domains | +1.24 |
| Extra prompt encoding | +0.10 |
| GPU memory pressure, Python closure overhead | ~+1.66 |
| **Total additional** | **+3.55 s** |
| **Original baseline** | **4.60 s** |
| **Predicted total** | **8.15 s** |
| **Measured total** | **8.15 s** |

#### 15.4 Summary

| Metric | Original | Ours | Delta |
|--------|----------|------|-------|
| Average generation time | 4.60 s | 8.15 s | +3.55 s (+77%) |
| Transformer forward passes — inversion | 14 | 15 (+extraction) | +1 |
| Transformer forward passes — denoising | 28 (mostly cached) | 28 + cfg_tail_steps extra uncached | +4 to +7 |
| Attention sequence length (inject blocks, early steps) | $S_\text{txt}+S_\text{img} = 1536$ | $S_\text{txt}+2S_\text{img} = 2560$ | +1024 tokens |
| Cache-bypassed block-steps | 0 | ref_inject_steps × ref_inject_blocks | up to 18×16=288 |

---

### 16. Design Rationale Summary

| Choice | Why |
|--------|-----|
| Pre-project ref K,V at extraction | Removes $2 \times \texttt{ref\_inject\_blocks} \times \texttt{ref\_inject\_steps}$ linear ops from the hot path |
| Sequential (not batched) CFG passes | `FluxTransformer2DModel_PREDEFINE` assumes batch=1; batching raises shape error in gate broadcast |
| Inversion detection via `_prev_step` drop | No pipeline modification required; `current['step']` state is observable from the patch closure |
| η=γ=0.75 universally (vs paper's domain-specific 0.5–1.0) | With ref_inject active, stronger inversion holds background against combined K,V + CFG pull. RR was 1.0→0.75 to allow FLUX lighting/shadow blending |
| RS stop_timestep=22 vs paper's 10 | Sketch backgrounds have minimal texture; without prolonged inversion injection the line-art structure dissolves. 22/28 steps keeps geometry grounded |
| RC/RP stop_timestep shortened (18→13, 15→12) | Frees FLUX to stylize once ref_inject handles foreground identity |
| Domain-specific cfg_scale (2.5–5.0) | Painting (5.0) needs strongest style push; same-domain RR (2.5) needs minimal guidance |
| bbox_margin=0.10 for RC/RP only | Tight dataset bboxes leave no blending room at edges in cross-domain compositing; RS/RR subjects have cleaner boundaries |
