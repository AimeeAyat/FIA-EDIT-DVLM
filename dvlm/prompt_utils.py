"""
Domain-aware prompt augmentation for EEdit composition.

The original EEdit prompts are typically short scene descriptions
(e.g. "a sheep in a cartoon field").  Augmenting them with explicit
style-integration and identity-preservation language significantly
improves FLUX's text-conditioned generation in cross-domain scenarios.

Negative-prompt templates are also provided for optional tail-CFG
(applying CFG only for the last few denoising steps to improve detail
without doubling total compute).
"""

# ── Per-domain positive-prompt suffixes ──────────────────────────────────────

_POS_SUFFIX = {
    "RC": (
        " The inserted object maintains its distinctive visual features, "
        "style-harmonised to blend naturally with the cartoon art style, "
        "smooth cel-shading, vibrant colours, clean outlines. "
        "Complete figure shown in full, untruncated, with all limbs and body parts visible."
    ),
    "RP": (
        " The inserted object is rendered with painterly brushstrokes "
        "consistent with the painting's style, textured surface, "
        "natural colour blending with the background. "
        "Complete figure shown in full, all parts of the object visible and naturally posed."
    ),
    "RS": (
        " The inserted object is drawn in the same pencil sketch or line-art "
        "style as the scene, clean outlines, minimal shading, monochrome. "
        "Complete figure, full body visible, no missing or cut-off parts."
    ),
    "RR": (
        " Photorealistic, high quality, consistent lighting and shadows, "
        "natural integration with the background scene. "
        "Complete figure shown in full, all body parts visible, naturally posed."
    ),
}

# ── Per-domain negative-prompt templates (used with optional tail-CFG) ────────

_NEG_PROMPT = {
    "RC": (
        "photorealistic, photo, out-of-style, mismatched style, "
        "artifacts, hard edge, blurry boundary, extra limbs, "
        "distorted proportions, low quality, "
        "truncated body, cut off, cropped figure, partial object, missing limbs, "
        "floating torso, incomplete figure, object behind obstruction"
    ),
    "RP": (
        "photorealistic, photo, sharp lines, out-of-style, "
        "artifacts, hard edge, low quality, distorted, "
        "truncated, cut off, cropped figure, partial body, missing parts, "
        "incomplete object, floating torso"
    ),
    "RS": (
        "coloured, photorealistic, watercolour, out-of-style, "
        "artifacts, blurry, low quality, extra lines, "
        "truncated, cut off, cropped, partial figure, missing limbs, "
        "incomplete sketch, floating torso"
    ),
    "RR": (
        "cartoon, painting, sketch, distorted, artifacts, "
        "low quality, blurry, watermark, text, extra limbs, "
        "truncated body, cut off, cropped, partial figure, missing body parts, "
        "floating torso, incomplete object"
    ),
}


def detect_domain(config_path: str) -> str:
    """
    Infer the domain code from the config file path.
    Returns one of "RC", "RP", "RS", "RR", or "RR" as default.
    """
    cp = config_path.lower()
    if "rc_" in cp or "real-cartoon" in cp or "realcartoon" in cp:
        return "RC"
    if "rp_" in cp or "real-painting" in cp or "realpainting" in cp:
        return "RP"
    if "rs_" in cp or "real-sketch" in cp or "realsketch" in cp:
        return "RS"
    if "rr_" in cp or "real-real" in cp or "realreal" in cp:
        return "RR"
    return "RR"   # safest default: same-domain behaviour


def augment_prompt(prompt: str, domain: str) -> str:
    """
    Append domain-specific guidance text to the user's prompt.
    Keeps the original prompt intact and adds the suffix only once.
    """
    suffix = _POS_SUFFIX.get(domain, "")
    if not suffix or suffix.strip() in prompt:
        return prompt
    return prompt.rstrip() + suffix


def get_negative_prompt(domain: str) -> str:
    """Return the recommended negative prompt for a given domain."""
    return _NEG_PROMPT.get(domain, _NEG_PROMPT["RR"])
