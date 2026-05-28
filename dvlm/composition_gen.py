"""
dvlm/composition_gen.py

Identical flow to the original composition_gen.py.
The only addition: domain-aware prompt augmentation via dvlm/prompt_utils.py.
"""

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _ROOT_DIR)

from cache_functions import *
from transformers import T5EncoderModel
from diffusers.utils import load_image
from MyCodes.MyFluxCompositionPipeline import FluxCompositionPipeline
import MyCodes.MyFluxForward as MyFluxForward
from MyCodes.myutils import seed_everything

from dvlm.prompt_utils import augment_prompt, get_negative_prompt, detect_domain

# Blackwell (sm_120) bfloat16 LayerNorm fix
import torch.nn as nn
import torch.nn.functional as _F
def _ln_fp32_forward(self, x):
    w = self.weight.float() if self.weight is not None else None
    b = self.bias.float()   if self.bias   is not None else None
    return _F.layer_norm(x.float(), self.normalized_shape, w, b, self.eps).to(x.dtype)
nn.LayerNorm.forward = _ln_fp32_forward


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
    nums = [int(f.split('.')[0].split('-')[-1]) for f in files
            if f.split('.')[0].split('-')[-1].isdigit()]
    return max(nums) + 1 if nums else 1


def parse_args():
    p = argparse.ArgumentParser(description='dvlm composition (prompt-only enhancements)')
    p.add_argument('--weights_dir',   type=str, required=True)
    p.add_argument('--config_path',   type=str, required=True)
    p.add_argument('--img_config',    type=str, required=True)
    p.add_argument('--output_dir',    type=str, default='test_outputs/dvlm_composition')
    p.add_argument('--use_predefine', type=bool, default=False)
    p.add_argument('--cpu_offload',   action='store_true')
    p.add_argument('--detect_nan',    action='store_true')
    p.add_argument('--no_prompt_aug', action='store_true',
                   help='Disable automatic prompt augmentation')
    p.add_argument('--use_tail_cfg', action='store_true',
                   help='Pass domain negative prompt to pipeline (tail-CFG)')
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
    pipe = FluxCompositionPipeline.from_pretrained(
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
        pipe.to('cuda')

    if args.detect_nan:
        _nan_found = [False]
        def _nan_hook(module, _inp, out):
            if _nan_found[0]:
                return
            outs = out if isinstance(out, (list, tuple)) else (out,)
            for o in outs:
                if isinstance(o, torch.Tensor) and torch.isnan(o).any():
                    _nan_found[0] = True
                    print(f"[NaN] first detected in: {module.__class__.__name__}")
                    import traceback; traceback.print_stack()
        for m in pipe.transformer.modules():
            m.register_forward_hook(_nan_hook)

    return pipe


def generate_image(pipe, img_config, param_config, output_dir, domain, args):
    main_image  = load_image(img_config["main_image"])
    ref_image   = load_image(img_config["ref_image"])
    ref_segment = load_image(img_config["ref_segment"])
    height = width = 512

    # RS domain: convert reference to monochrome so it matches the sketch scene
    if domain == "RS":
        ref_image = ref_image.convert("L").convert("RGB")

    prompt = img_config["prompt"]
    if not args.no_prompt_aug:
        prompt = augment_prompt(prompt, domain)
        print(f"[prompt] {prompt[:120]}...")

    for param in param_config['params']:
        cache_type     = param.get('cache_type', 'ours_predefine')
        ratio_scheduler = 'constant'
        use_attn_map   = False

        model_kwargs = {
            'fresh_ratio':       param['fresh_ratio'],
            'cache_type':        cache_type,
            'ratio_scheduler':   ratio_scheduler,
            'force_fresh':       'global',
            'fresh_threshold':   param['fresh_threshold'],
            'soft_fresh_weight': param['soft_fresh_weight'],
            'tailing_step':      param['tailing_step'],
            'edit_base':         2,
            'hw':                (height // 16, width // 16),
        }

        edit_idx = None if param['cascade_num'] == 0 else edit_region_parser(
            img_config['x1'], img_config['y1'],
            img_config['x2'], img_config['y2'],
            cascade_num=param['cascade_num'],
            height=height, width=width,
        )
        cache_dic, current = cache_init(model_kwargs, param['num_inference_steps'], edit_idx)
        current['edit_idx_merged'] = convert_to_cache_index(
            edit_idx, edit_base=2, bonus_ratio=0.8, height=height, width=width
        )
        current['edit_idx_merged'] = current['edit_idx_merged'].to("cuda")
        if cache_type == 'ours_predefine':
            predefine_cache_fresh_indices(cache_dic, current)

        joint_attention_kwargs = {
            'use_attn_map': use_attn_map,
            'cache_dic':    cache_dic,
            'use_cache':    param['use_cache'],
            'current':      current,
        }

        torch.manual_seed(42)
        t0 = time.time()
        neg_prompt = get_negative_prompt(domain) if args.use_tail_cfg else img_config.get("neg_prompt", None)

        res = pipe.gen(
            prompt=prompt,
            neg_prompt=neg_prompt,
            main_image=main_image,
            ref_image=ref_image,
            ref_segment=ref_segment,
            height=512,
            width=512,
            x1=img_config["x1"], y1=img_config["y1"],
            x2=img_config["x2"], y2=img_config["y2"],
            num_inference_steps=param['num_inference_steps'],
            guidance_scale=param.get('guidance_scale', 7.0),
            joint_attention_kwargs=joint_attention_kwargs,
            use_rf_inversion=param['use_rf_inversion'],
            eta=param['eta'],
            gamma=param['gamma'],
            start_timestep=param['start_timestep'],
            stop_timestep=param['stop_timestep'],
            blend_ratio=param['blend_ratio'],
            generator=torch.Generator(device='cuda').manual_seed(42),
            skip_T=param.get('inv_skip', 3),
        )
        elapsed = time.time() - t0

        image = res.images[0]
        num   = get_next_number(output_dir)
        image.save(f"{output_dir}/{num:03d}.png")

        timing_path = os.path.join(output_dir, "timing.json")
        timings = json.load(open(timing_path)) if os.path.exists(timing_path) else []
        timings.append({
            "image": f"{num:03d}.png", "seconds": round(elapsed, 2),
            "steps": param['num_inference_steps'],
            "cache_type": cache_type, "use_cache": param.get('use_cache', False),
            "domain": domain,
        })
        json.dump(timings, open(timing_path, 'w'), indent=2)
        print(f"[timing] {num:03d}.png — {elapsed:.1f}s  domain={domain}")


def main():
    args   = parse_args()
    domain = detect_domain(args.config_path)
    print(f"[dvlm] Domain: {domain}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    pipe = load_models(args)

    with open(args.img_config) as f:
        img_configs = json.load(f)
    with open(args.config_path) as f:
        param_config = json.load(f)

    seed_everything()
    for img_config in img_configs['imgs']:
        generate_image(pipe, img_config, param_config, args.output_dir, domain, args)


if __name__ == "__main__":
    main()
