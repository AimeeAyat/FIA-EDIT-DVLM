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
#     "RC": (
#         " Naturally placed in the scene with correct ground contact, "
#         "consistent cast shadow, proper depth and scale relative to the background. "
#         "The object retains its original appearance and is not re-drawn or re-styled."
#     ),
#     "RP": (
#         " Naturally placed in the scene with correct ground contact, "
#         "consistent cast shadow, proper depth and scale relative to the background. "
#         "The object retains its original appearance and is not re-drawn or re-styled."
#     ),
#     "RS": (
#         " Naturally placed in the scene with correct ground contact, "
#         "consistent cast shadow, proper depth and scale relative to the background. "
#         "The object retains its shape and style but looks hand sketch in the scene."
#     ),
#     "RR": (
#         " Naturally placed in the scene with correct ground contact, "
#         "consistent lighting and cast shadow, proper depth and scale. "
#         "High quality, photorealistic integration."
#     ),
# }

# # ── Per-domain negative-prompt templates ─────────────────────────────────────

# _NEG_PROMPT = {
#     "RC": (
#         "artifacts, hard edge, blurry boundary, distorted proportions, low quality, "
#         "truncated body, cut off, partial figure, missing limbs, floating object"
#     ),
#     "RP": (
#         "artifacts, hard edge, low quality, distorted, "
#         "truncated, cut off, partial body, floating object"
#     ),
#     "RS": (
#         "artifacts, blurry, low quality, "
#         "truncated, cut off, partial figure, floating object"
#     ),
#     "RR": (
#         "distorted, artifacts, low quality, blurry, watermark, text, "
#         "truncated body, cut off, partial figure, floating object"
#     ),
# }

_POS_SUFFIX = {
    "RC": (
        "Naturally integrated into the scene with realistic ground contact, accurate cast shadows, "
        "correct perspective, depth, and scale relative to the background. "
        "The reference subject retains its original identity, facial features, clothing, and proportions, "
        "but is rendered in a vibrant, consistent cartoon style that matches the overall scene. "
        "Seamless blending, no visible cut-out or pasting artifacts."
    ),
    "RP": (
        "Naturally placed in the scene with proper ground contact, soft cast shadows, "
        "coherent artistic lighting, correct depth and scale. "
        "The reference subject keeps exact identity, face, pose, and details but is rendered "
        "in a high-quality painterly/oil-painting style consistent with the background. "
        "Visible brush strokes and artistic texture, yet perfectly merged."
    ),
    "RS": (
        "Naturally embedded in the scene with correct grounding, cast shadows, and perspective. "
        "The reference subject maintains identical shape, face, features, proportions, and pose as the reference, "
        "but is represented as a clean, high-quality sketch/line drawing with hatching and line work "
        "that harmonizes with the background. Artistic sketch style, not photorealistic."
    ),
    "RR": (
        "Photorealistically integrated into the scene with perfect ground contact, realistic cast shadows, "
        "consistent lighting, accurate perspective, depth, and scale. "
        "The reference subject retains its exact original appearance, details, and photorealism. "
        "High fidelity, seamless composition, indistinguishable from a real photograph."
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
