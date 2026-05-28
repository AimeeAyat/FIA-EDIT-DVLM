"""
dvlm/composition_gen.py — enhanced drop-in replacement for composition_gen.py

Usage (same interface as original):
    python dvlm/composition_gen.py \
        --weights_dir ./weights \
        --config_path ./dvlm/domain_configs/RC_config.json \
        --img_config  ./configs/composition/Real-Cartoon.json \
        --output_dir  ./EEdit_outputs/dvlm/Real-Cartoon \
        --use_predefine 1

New optional flags:
    --domain      RC | RP | RS | RR  (auto-detected from config_path if omitted)
    --no_seamless                    disable seamless composite blending
    --expand_bbox FLOAT              expand bounding box by this fraction (default 0.10)
    --no_ref_inject                  disable reference token attention injection
    --no_prompt_aug                  disable automatic prompt augmentation
    --use_tail_cfg                   enable tail-CFG with negative prompt (last 5 steps)
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import torch

# ── sys path setup so we can import from both the EEdit root and dvlm/ ────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

from cache_functions import *
from transformers import T5EncoderModel
from diffusers.utils import load_image
import MyCodes.MyFluxForward as MyFluxForward
from MyCodes.myutils import seed_everything

from dvlm.enhanced_pipeline import EnhancedFluxCompositionPipeline
from dvlm.prompt_utils import augment_prompt, detect_domain


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_transformer_config(weights_dir):
    import shutil
    dir_path = os.path.join(weights_dir, "transformer")
    if os.path.isfile(os.path.join(dir_path, "config.json")):
        return dir_path
    fallback_dir = os.path.join(weights_dir, "transformer_config_dir")
    os.makedirs(fallback_dir, exist_ok=True)
    dst = os.path.join(fallback_dir, "config.json")
    if not os.path.exists(dst):
        shutil.copy2(os.path.join(weights_dir, "transformer_config.json"), dst)
    return fallback_dir


def get_next_number(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    files = [f for f in os.listdir(dirname)]
    if not files:
        return 1
    nums = [
        int(f.split(".")[0].split("-")[-1])
        for f in files
        if f.split(".")[0].split("-")[-1].isdigit()
    ]
    return max(nums) + 1 if nums else 1


def parse_args():
    p = argparse.ArgumentParser(description="Enhanced EEdit composition (dvlm)")
    p.add_argument("--weights_dir",   type=str, required=True)
    p.add_argument("--config_path",   type=str, required=True)
    p.add_argument("--img_config",    type=str, required=True)
    p.add_argument("--output_dir",    type=str, default="test_outputs/dvlm_composition")
    p.add_argument("--use_predefine", type=bool, default=False)
    p.add_argument("--cpu_offload",   action="store_true")
    # Enhancement controls
    p.add_argument("--domain",        type=str, default=None,
                   help="RC|RP|RS|RR — auto-detected if omitted")
    p.add_argument("--no_seamless",   action="store_true",
                   help="Disable Poisson/seamless composite blending")
    p.add_argument("--expand_bottom",  type=float, default=0.18,
                   help="Expand bbox downward by this fraction (for hidden legs/feet)")
    p.add_argument("--expand_top",     type=float, default=0.0,
                   help="Expand bbox upward (rarely needed)")
    p.add_argument("--expand_sides",   type=float, default=0.0,
                   help="Expand bbox left/right — keep 0 to avoid including background")
    p.add_argument("--no_ref_inject",        action="store_true",
                   help="Disable reference token attention injection")
    p.add_argument("--no_prompt_aug",        action="store_true",
                   help="Disable automatic prompt augmentation")
    p.add_argument("--use_tail_cfg",         action="store_true",
                   help="Enable tail-CFG (last 5 steps) with negative prompt")
    # Colour harmonisation
    p.add_argument("--no_color_harmonize",   action="store_true",
                   help="Disable Reinhard colour transfer before paste")
    return p.parse_args()


def load_models(args, dtype=torch.bfloat16):
    if args.use_predefine:
        from MyCodes.FluxTransformer2DModel_PREDEFINE import FluxTransformer2DModel
    else:
        from MyCodes.FluxTransformer2DModel import FluxTransformer2DModel

    transformer = FluxTransformer2DModel.from_single_file(
        pretrained_model_link_or_path_or_dict=f"{args.weights_dir}/flux1-dev.safetensors",
        config=_resolve_transformer_config(args.weights_dir),
        torch_dtype=dtype,
        local_files_only=True,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.weights_dir, subfolder="text_encoder_2", torch_dtype=dtype
    )
    # Load as EnhancedFluxCompositionPipeline
    pipe = EnhancedFluxCompositionPipeline.from_pretrained(
        args.weights_dir,
        transformer=None,
        text_encoder_2=None,
        torch_dtype=dtype,
    )
    pipe.transformer    = transformer
    pipe.text_encoder_2 = text_encoder_2
    pipe.transformer.forward = types.MethodType(MyFluxForward.forward, pipe.transformer)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    if args.cpu_offload or vram_gb < 40:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe


def generate_image(pipe, img_config, param_config, output_dir, args, domain):
    main_image   = load_image(img_config["main_image"])
    ref_image    = load_image(img_config["ref_image"])
    ref_segment  = load_image(img_config["ref_segment"])
    height = width = 512

    # Optionally augment the prompt
    prompt = img_config["prompt"]
    if not args.no_prompt_aug:
        prompt = augment_prompt(prompt, domain)
        print(f"[prompt] {prompt[:120]}...")

    for param in param_config["params"]:
        cache_type = "ours_predefine"
        if "cache_type" in param:
            cache_type = param["cache_type"]
        ratio_scheduler = "constant"
        use_attn_map    = False

        model_kwargs = {
            "fresh_ratio":       param["fresh_ratio"],
            "cache_type":        cache_type,
            "ratio_scheduler":   ratio_scheduler,
            "force_fresh":       "global",
            "fresh_threshold":   param["fresh_threshold"],
            "soft_fresh_weight": param["soft_fresh_weight"],
            "tailing_step":      param["tailing_step"],
            "edit_base":         2,
            "hw":                (height // 16, width // 16),
        }

        edit_idx = None if param["cascade_num"] == 0 else edit_region_parser(
            img_config["x1"], img_config["y1"],
            img_config["x2"], img_config["y2"],
            cascade_num=param["cascade_num"],
            height=height, width=width,
        )
        cache_dic, current = cache_init(model_kwargs, param["num_inference_steps"], edit_idx)
        current["edit_idx_merged"] = convert_to_cache_index(
            edit_idx, edit_base=2, bonus_ratio=0.8, height=height, width=width
        )
        current["edit_idx_merged"] = current["edit_idx_merged"].to("cuda")

        if cache_type == "ours_predefine":
            predefine_cache_fresh_indices(cache_dic, current)

        joint_attention_kwargs = {
            "use_attn_map": use_attn_map,
            "cache_dic":    cache_dic,
            "use_cache":    param["use_cache"],
            "current":      current,
        }

        torch.manual_seed(42)
        t0 = time.time()

        res = pipe.gen(
            prompt=prompt,
            main_image=main_image,
            ref_image=ref_image,
            ref_segment=ref_segment,
            height=height,
            width=width,
            x1=img_config["x1"], y1=img_config["y1"],
            x2=img_config["x2"], y2=img_config["y2"],
            num_inference_steps=param["num_inference_steps"],
            joint_attention_kwargs=joint_attention_kwargs,
            use_rf_inversion=param["use_rf_inversion"],
            eta=param["eta"],
            gamma=param["gamma"],
            start_timestep=param["start_timestep"],
            stop_timestep=param["stop_timestep"],
            blend_ratio=param["blend_ratio"],
            generator=torch.Generator(device="cuda").manual_seed(42),
            skip_T=param.get("inv_skip", 3),
            # Enhancement args
            ref_inject_blocks=0 if args.no_ref_inject else param.get("ref_inject_blocks", 8),
            ref_inject_steps=param.get("ref_inject_steps", 14),
            use_seamless_blend=not args.no_seamless,
            expand_bottom_frac=args.expand_bottom,
            expand_top_frac=args.expand_top,
            expand_sides_frac=args.expand_sides,
            domain=domain,
            preprocess_ref=(not args.no_prompt_aug),
            use_tail_cfg=args.use_tail_cfg,
            cfg_tail_steps=param.get("cfg_tail_steps", 5),
            cfg_scale=param.get("cfg_scale", 3.5),
            # Colour harmonisation
            color_harmonize=not args.no_color_harmonize,
        )
        elapsed = time.time() - t0

        image = res.images[0]
        num   = get_next_number(output_dir)
        image.save(f"{output_dir}/{num:03d}.png")

        timing_path = os.path.join(output_dir, "timing.json")
        timings = json.load(open(timing_path)) if os.path.exists(timing_path) else []
        timings.append({
            "image":      f"{num:03d}.png",
            "seconds":    round(elapsed, 2),
            "steps":      param["num_inference_steps"],
            "cache_type": param.get("cache_type", "none"),
            "use_cache":  param.get("use_cache", False),
            "domain":     domain,
            "ref_inject": not args.no_ref_inject,
            "seamless":   not args.no_seamless,
        })
        json.dump(timings, open(timing_path, "w"), indent=2)
        print(f"[timing] {num:03d}.png — {elapsed:.1f}s  domain={domain}  "
              f"ref_inject={not args.no_ref_inject}  seamless={not args.no_seamless}")


def main():
    args = parse_args()

    domain = args.domain or detect_domain(args.config_path)
    print(f"[dvlm] Domain: {domain}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pipe = load_models(args)

    with open(args.img_config) as f:
        img_configs = json.load(f)
    with open(args.config_path) as f:
        param_config = json.load(f)

    seed_everything()

    for img_config in img_configs["imgs"]:
        generate_image(pipe, img_config, param_config, args.output_dir, args, domain)


if __name__ == "__main__":
    main()
