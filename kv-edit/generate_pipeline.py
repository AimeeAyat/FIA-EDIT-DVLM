import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(20, 11))
ax.set_xlim(0, 20)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

# ── colour palette ────────────────────────────────────────────────────────────
C_INPUT   = '#D6EAF8'   # light blue – inputs
C_MODULE  = '#D5F5E3'   # light green – processing modules
C_ATTN    = '#FDEBD0'   # light orange – attention / injection
C_OUTPUT  = '#F9EBEA'   # light red – output
C_ARROW   = '#2C3E50'
C_TITLE   = '#1A252F'

def box(ax, x, y, w, h, label, sublabel='', color='#EFEFEF',
        fontsize=10, subfontsize=8):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0.08',
                          linewidth=1.4, edgecolor='#555',
                          facecolor=color, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x + w/2, y + h*0.62, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=C_TITLE, zorder=4)
        ax.text(x + w/2, y + h*0.28, sublabel, ha='center', va='center',
                fontsize=subfontsize, color='#555', zorder=4, style='italic')
    else:
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=C_TITLE, zorder=4,
                multialignment='center')

def arrow(ax, x0, y0, x1, y1, label='', color=C_ARROW, lw=1.6):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, connectionstyle='arc3,rad=0'))
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.08, my+0.12, label, fontsize=7.5, color='#444',
                ha='center', zorder=5)

def dashed_arrow(ax, x0, y0, x1, y1, color='#777', lw=1.4):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle='dashed',
                                connectionstyle='arc3,rad=0'))

# ── TITLE ─────────────────────────────────────────────────────────────────────
ax.text(10, 10.55, 'Flux & Key: Reference-Guided Image Editing Pipeline',
        ha='center', va='center', fontsize=15, fontweight='bold', color=C_TITLE)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 – Inputs
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 0.3,  8.5, 2.4, 1.1, 'Source Image\n$\\mathbf{I}_A$', color=C_INPUT)
box(ax, 3.2,  8.5, 2.4, 1.1, 'Reference Image\n$\\mathbf{I}_B$', color=C_INPUT)
box(ax, 6.1,  8.5, 2.4, 1.1, 'Edit Mask\n$\\mathbf{M}_A$', color=C_INPUT)
box(ax, 9.0,  8.5, 2.4, 1.1, 'Text Prompt\n$\\mathcal{T}$', color=C_INPUT)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 – Stage modules
# ══════════════════════════════════════════════════════════════════════════════

# Stage 1: Reference extraction
box(ax, 2.6, 6.6, 3.2, 1.2,
    'Stage 1: Reference\nObject Extraction',
    'Grounded-DINO + SAM 2', color=C_MODULE, subfontsize=8)

# Stage 2: VAE encode
box(ax, 0.3, 6.6, 1.9, 1.2,
    'VAE Encode\n$\\mathcal{E}(\\mathbf{I}_A)$',
    color=C_MODULE, fontsize=9)

# Stage 3: Source features
box(ax, 0.3, 4.9, 1.9, 1.2,
    'Src Features\n(1 fwd pass)',
    '$\\{K_{\\mathrm{src}}, V_{\\mathrm{src}}\\}$', color=C_MODULE, subfontsize=8)

# SSI
box(ax, 6.1, 6.6, 2.4, 1.2,
    'Stage 3: SSI',
    '$\\tilde{z}_0^{\\mathcal{F}} = \\sqrt{1-\\tau^2}\\,z_A^{\\mathcal{F}} + \\tau\\varepsilon$',
    color=C_MODULE, subfontsize=7.5)

# Reference aligned
box(ax, 2.6, 4.9, 3.2, 1.2,
    'Aligned Ref Latent\n$\\mathbf{z}_B = \\mathcal{E}(\\mathbf{I}_B^{\\mathrm{aligned}})$',
    'RoPE re-indexed to $\\mathcal{F}$', color=C_MODULE, subfontsize=7.5)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 – Core attention block (centre piece)
# ══════════════════════════════════════════════════════════════════════════════

# Outer frame for the attention block
outer = FancyBboxPatch((0.15, 2.3), 11.4, 2.3,
                       boxstyle='round,pad=0.12',
                       linewidth=2, edgecolor='#888',
                       facecolor='#F0F0F0', zorder=2)
ax.add_patch(outer)
ax.text(5.85, 4.53, 'Stage 4 & 5 — Spectral-Aware Dual-Source KV Injection  (per block $\\ell$)',
        ha='center', va='center', fontsize=9, color='#333', style='italic')

# Background attention
box(ax, 0.4, 2.5, 3.0, 1.7,
    'Background Tokens\n$\\mathcal{B}$',
    'Freq-Locked:\n'
    '$K_{\\mathcal{B}}=\\mathcal{F}^{-1}(\\mathcal{H}(K_{\\mathrm{src}})+\\mathcal{L}(K_{\\mathrm{tgt}}))$',
    color='#EAF2F8', fontsize=9, subfontsize=7)

# Foreground attention
box(ax, 4.2, 2.5, 4.2, 1.7,
    'Foreground Tokens $\\mathcal{F}$',
    'SA-DSKV:\n'
    '$\\tilde{K}=\\mathcal{F}^{-1}(\\alpha\\mathcal{F}(K_{\\mathrm{text}})+(1-\\alpha)\\mathcal{F}(K_{\\mathrm{ref}}))$\n'
    'schedule: $\\alpha^{(\\ell)}=\\alpha_{\\max}-\\frac{\\ell}{L}(\\alpha_{\\max}-\\alpha_{\\min})$',
    color=C_ATTN, fontsize=9, subfontsize=6.8)

# No cross
ax.text(3.7, 3.35, '✕', ha='center', va='center', fontsize=18,
        color='#c0392b', fontweight='bold', zorder=5)
ax.text(3.7, 2.9, 'No mixing', ha='center', va='center',
        fontsize=7, color='#c0392b', zorder=5)

# Denoising loop annotation
box(ax, 9.0, 2.5, 2.5, 1.7,
    'Denoising Loop\n$t = 1,\\ldots,T$',
    'FLUX DiT\n($L$ blocks)', color='#EDE7F6', fontsize=9, subfontsize=8)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 – Post processing & output
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 3.5, 0.5, 3.0, 1.2,
    'VAE Decode\n$\\mathcal{D}(z_{\\mathrm{final}})$', color=C_MODULE)
box(ax, 7.3, 0.5, 3.0, 1.2,
    'Histogram\nHarmonisation', color=C_MODULE)
box(ax, 11.1, 0.5, 3.0, 1.2,
    'Edited Image\n$\\hat{\\mathbf{I}}$', color=C_OUTPUT)

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS
# ══════════════════════════════════════════════════════════════════════════════
# Inputs → stages
arrow(ax, 1.5,  8.5, 1.5,  7.8)           # I_A → VAE encode
arrow(ax, 4.4,  8.5, 4.4,  7.8)           # I_B → Stage 1
arrow(ax, 7.3,  8.5, 7.3,  7.8)           # M_A → SSI (also Stage 1)
arrow(ax, 10.2, 8.5, 10.2, 4.2,           # T → denoising loop (long)
      color='#7F8C8D')
# I_B also goes to Stage 1
arrow(ax, 3.7, 8.5, 3.7, 7.8)
# M_A → Stage 1 (dashed hint)
dashed_arrow(ax, 6.5, 8.5, 5.5, 7.8)

# Stage 1 → aligned ref
arrow(ax, 4.2, 6.6, 4.2, 6.1)
# VAE encode → src features
arrow(ax, 1.3, 6.6, 1.3, 6.1)
# VAE encode → SSI
arrow(ax, 2.1, 7.2, 6.1, 7.2)

# SSI → foreground tokens
arrow(ax, 7.3, 6.6, 6.6, 4.2)

# Src features → background tokens (freq lock)
arrow(ax, 1.3, 4.9, 1.3, 4.2, '$K_{\\mathrm{src}},V_{\\mathrm{src}}$')

# Aligned ref → foreground tokens
arrow(ax, 4.2, 4.9, 5.5, 4.2, '$K_{\\mathrm{ref}},V_{\\mathrm{ref}}$')

# Denoising → decode
arrow(ax, 10.2, 2.5, 6.5, 1.7)

# Decode → harmonise → output
arrow(ax, 6.5, 1.1, 7.3, 1.1)
arrow(ax, 10.3, 1.1, 11.1, 1.1)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_x, legend_y = 13.2, 7.0
ax.text(legend_x, legend_y + 1.0, 'Legend', fontsize=10,
        fontweight='bold', color=C_TITLE)
for color, label in [(C_INPUT,  'Input'),
                     (C_MODULE, 'Processing Module'),
                     (C_ATTN,   'Attention Injection'),
                     (C_OUTPUT, 'Output')]:
    p = FancyBboxPatch((legend_x, legend_y - 0.05), 0.55, 0.42,
                       boxstyle='round,pad=0.05',
                       linewidth=1, edgecolor='#777', facecolor=color)
    ax.add_patch(p)
    ax.text(legend_x + 0.72, legend_y + 0.16, label,
            fontsize=8.5, va='center', color='#333')
    legend_y -= 0.65

# ── note on frequency split ───────────────────────────────────────────────────
ax.text(13.2, 4.0,
        '$\\mathcal{H}$: high-freq ($\\|{(u,v)}\\|>\\gamma$)\n'
        '$\\mathcal{L}$: low-freq ($\\|{(u,v)}\\|\\leq\\gamma$)\n'
        '$\\mathcal{F}$: 2-D DFT\n'
        'SSI: Selective Stochastic Inversion\n'
        'SA-DSKV: Spectral-Aware Dual-Source KV',
        fontsize=7.8, va='top', color='#444',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                  edgecolor='#CCC', linewidth=1))

plt.tight_layout(pad=0.3)
plt.savefig('methodology_pipeline.pdf', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.savefig('methodology_pipeline.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print('Saved: methodology_pipeline.pdf and methodology_pipeline.png')
