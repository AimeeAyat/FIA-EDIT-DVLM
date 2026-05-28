"""
Domain-aware prompt augmentation for EEdit composition.

Goal: guide FLUX toward natural scene integration (shadows, ground contact,
depth) WITHOUT changing the reference object's appearance, colours or texture.
Style-transformation language is intentionally omitted — the reference identity
must be preserved as-is.
"""

# ── Per-domain positive-prompt suffixes ──────────────────────────────────────
# Integration-only: scene coherence, shadows, ground contact.
# No style-transfer language that would cause FLUX to re-generate the object.


# _POS_SUFFIX = {
# "RC": (
    
#     "Cartoon style. Seamless integration into the scene. "
#     "Matching linework, cel-shading, lighting, shadows, perspective, and scale. "
#     "Original face, identity, and proportions preserved."
# ),

# "RP": (
#     "Painterly style. Seamless integration into the scene. "
#     "Matching brushwork, texture, lighting, shadows, perspective, and scale. "
#     "Original face, identity, and proportions preserved."
# ),

# "RS": (
#     "Sketch style. Seamless integration into the scene. "
#     "Matching linework, hatching, shading, perspective, and scale. Non-photorealistic. "
#     "Original face, identity, and proportions preserved."
# ),

# "RR": (
#     "Photorealistic style. Seamless integration into the scene. "
#     "Matching lighting, shadows, textures, perspective, and scale. "
#     "Original face, identity, and proportions preserved."
# ),
# }
_POS_SUFFIX = {
"RC": (
    
    "cartoon style, bold thick black outlines, flat vector colors, fun, exaggerated expressions and proportions."
    "Original face, identity, and proportions preserved."
),

"RP": (
    "realistic painting, featuring visible textures, artistic lighting, and authentic paint strokes."
    "Original face, identity, and proportions preserved."
),

"RS": (
    "Sketch style. hand-drawn, featuring realistic line work, manual shading, and authentic paper texture."
    "Original face, identity, and proportions preserved."
),

"RR": (
    "Photorealistic style. Seamless integration into the scene. "
    "Matching lighting, shadows, textures, perspective, and scale. "
    "Original face, identity, and proportions preserved."
),
}
_NEG_PROMPT = {
    "RC": (
        "artifacts, hard edges, blurry boundaries, cut-out look, pasted appearance, floating object, "
        "distorted proportions, deformed anatomy, extra limbs, missing limbs, low quality, cartoon style mismatch, "
        "photorealistic patches, inconsistent line thickness"
    ),
    "RP": (
        "artifacts, hard edges, cut-out, pasted, floating, distorted, low quality, "
        "photorealistic details, flat colors, pixelation, style mismatch, overly smooth skin"
    ),
    "RS": (
        "artifacts, blurry, hard edges, cut-out, floating object, realistic shading, photorealistic textures, "
        "thick unnatural lines, low quality sketch, filled solid areas, inconsistent line art"
    ),
    "RR": (
        "artifacts, distorted, blurry, watermark, text, hard edges, cut-out appearance, pasted look, "
        "floating, style leakage, cartoonish, painterly, sketch lines, low resolution, bad anatomy"
    ),
}

def detect_domain(config_path: str) -> str:
    cp = config_path.lower()
    if "rc_" in cp or "real-cartoon" in cp or "realcartoon" in cp:
        return "RC"
    if "rp_" in cp or "real-painting" in cp or "realpainting" in cp:
        return "RP"
    if "rs_" in cp or "real-sketch" in cp or "realsketch" in cp:
        return "RS"
    if "rr_" in cp or "real-real" in cp or "realreal" in cp:
        return "RR"
    return "RR"


def augment_prompt(prompt: str, domain: str) -> str:
    suffix = _POS_SUFFIX.get(domain, "")
    if not suffix or suffix.strip() in prompt:
        return prompt
    return prompt.rstrip() + suffix


def get_negative_prompt(domain: str) -> str:
    return _NEG_PROMPT.get(domain, _NEG_PROMPT["RR"])
