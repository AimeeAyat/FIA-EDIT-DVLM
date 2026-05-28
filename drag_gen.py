import argparse
from pathlib import Path
import torch
import json
from region_utils.drag import get_drag_data
from cache_functions import cache_init, edit_pts_parser,convert_to_cache_index
from MyCodes.MyFluxDragEditPipeline import FluxDragEditPipeline
from transformers import T5EncoderModel
from diffusers.utils import load_image
from MyCodes import MyFluxForward
import os
import types

# On Blackwell (sm_120) + PyTorch 2.11, bfloat16 LayerNorm variance underflows to
# negative -> sqrt(negative) = NaN. Patch to compute in float32 (input+weight+bias).
import torch.nn as nn
import torch.nn.functional as _F
def _ln_fp32_forward(self, x):
    w = self.weight.float() if self.weight is not None else None
    b = self.bias.float() if self.bias is not None else None
    return _F.layer_norm(x.float(), self.normalized_shape, w, b, self.eps).to(x.dtype)
nn.LayerNorm.forward = _ln_fp32_forward

def get_next_number(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    files = [f for f in os.listdir(dirname)]
    if not files:
        return 1
    nums = [int(f.split('.')[0].split('-')[-1]) for f in files if f.split('.')[0].split('-')[-1].isdigit()]
    return max(nums) + 1 if nums else 1


def parse_args():
    parser = argparse.ArgumentParser(description='code for drag')
    parser.add_argument('--weights_dir', type=str, default='/root/your-path/weights',
                       help='model weights directory')
    parser.add_argument('--config_path', type=str, 
                       default='/root/your-path/configs/drag/example_config.json',
                       help='path of config file')
    parser.add_argument('--img_config', type=str,
                       default='/root/your-path/configs/drag/example_imgs.json', 
                       help='path of image config file')
    parser.add_argument('--output_dir', type=str,
                       default='/root/your-path/test_outputs/drag',
                       help='output directory')
    parser.add_argument('--use_predefine', type=bool,
                       default=False,
                       help='whether to use predefine')
    parser.add_argument('--cpu_offload', action='store_true',
                       help='use enable_model_cpu_offload instead of pipe.to(cuda)')
    return parser.parse_args()

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

def load_models(args, dtype=torch.bfloat16):
    if args.use_predefine:
        from MyCodes.FluxTransformer2DModel_PREDEFINE import FluxTransformer2DModel
    else:
        from MyCodes.FluxTransformer2DModel import FluxTransformer2DModel
    transformer = FluxTransformer2DModel.from_single_file(
        pretrained_model_link_or_path_or_dict=f"{args.weights_dir}/flux1-dev.safetensors",
        config=_resolve_transformer_config(args.weights_dir),
        torch_dtype=dtype,
        local_files_only=True)

    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.weights_dir,
        subfolder="text_encoder_2",
        torch_dtype=dtype)

    pipe = FluxDragEditPipeline.from_pretrained(
        args.weights_dir,
        transformer=None,
        text_encoder_2=None,
        torch_dtype=dtype)
    pipe.transformer = transformer
    pipe.text_encoder_2 = text_encoder_2

    pipe.transformer.forward = types.MethodType(MyFluxForward.forward, pipe.transformer)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if args.cpu_offload or vram_gb < 40:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to('cuda')

    return pipe

def generate_image(pipe, img_config, param_config, output_dir):
    drag_data = get_drag_data(img_config['data_path'])
    main_image = load_image(os.path.join(img_config['data_path'], "original_image.png"))
    height = main_image.height
    width = main_image.width
    target_pts = drag_data['target']
    source_pts = drag_data['source']

    
    for param in param_config['params']:
        if 'cache_type' in param:
            ratio_scheduler = 'constant'
            use_attn_map=False
            if param['cache_type'] == 'ours_cache':
                cache_type = 'ours_cache'
            elif param['cache_type'] == 'ours_predefine':
                cache_type = 'ours_predefine'
          
            
        model_kwargs = {
            'fresh_ratio': param['fresh_ratio'],
            'cache_type': cache_type,
            'ratio_scheduler': ratio_scheduler,
            'force_fresh': 'global',
            'fresh_threshold': param['fresh_threshold'],
            'soft_fresh_weight': param['soft_fresh_weight'],
            'tailing_step': param['tailing_step'],
            'hw': (height//pipe.vae_scale_factor,width//pipe.vae_scale_factor)
        }
        edit_idx=edit_pts_parser(source_pts,target_pts,cascade_num=param['cascade_num'],height=height,width=width)
        cache_dic, current = cache_init(model_kwargs, param['num_inference_steps'],edit_idx)
        current['edit_idx_merged']=convert_to_cache_index(edit_idx,edit_base=2,bonus_ratio=0.8,height=height,width=width)
        current['edit_idx_merged']=current['edit_idx_merged'].to("cuda")
        joint_attention_kwargs = {
            'use_attn_map': use_attn_map,
            'cache_dic': cache_dic,
            'use_cache': param['use_cache'],
            'current': current,
        }
        torch.manual_seed(42)
        res = pipe.gen(
            prompt=drag_data['prompt'],
            main_image=main_image,
            src_region=None,
            tgt_region=None,
            src_pts=source_pts,
            tgt_pts=target_pts,
            strength=param['strength'],
            num_inference_steps=param['num_inference_steps'],
            eta=param['eta'],
            height=height,
            width=width,
            drag_class=param['drag_class'],
            t_prime_ratio=param['t_prime_ratio'],
            alpha=param['alpha'],
            gamma=param['gamma'],
            start_timestep=param['start_timestep'],
            stop_timestep=param['stop_timestep'],
            joint_attention_kwargs=joint_attention_kwargs,
            generator=torch.Generator(device='cuda').manual_seed(42),
            skip_T=3 if 'inv_skip' not in param else param['inv_skip']
        )
        image=res.images[0]
        num = get_next_number(output_dir)
        image.save(f"{output_dir}/{num:03d}.png")
        
    


def main():
    args = parse_args()
    
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    
    pipe= load_models(args)
    
    with open(args.img_config, 'r') as f:
        img_configs = json.load(f)
    with open(args.config_path, 'r') as f:
        param_config = json.load(f)
    for img_config in img_configs['imgs']:
        generate_image(pipe, img_config, param_config, args.output_dir)
        
if __name__ == "__main__":
    main()