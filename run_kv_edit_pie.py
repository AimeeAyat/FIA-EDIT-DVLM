import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from models.kv_edit import Flux_kv_edit,Flux_kv_edit_inf
from dataclasses import dataclass
from dataset.pie_bench.pie_features import PIE_features_Dataset
from tqdm import tqdm
import argparse
import os
import json
import gc

@dataclass
class SamplingOptions:
    # prompt: str
    width: int = 512
    height: int = 512
    inversion_num_steps: int = 0
    denoise_num_steps: int = 0
    skip_step: int = 0
    inversion_guidance: float = 1.0
    denoise_guidance: float = 1.0
    re_init: bool = False
    attn_mask: bool = False
    attn_scale: float = 0.0

def custom_collate(batch):
    
    return batch[0]


def run_inference(args):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    rank = 0
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'args.json'), 'w+') as f:
        json.dump(vars(args), f, indent=4)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = device
    opts = SamplingOptions(
            width=args.width,
            height=args.height,
            inversion_num_steps=args.inversion_num_steps,
            denoise_num_steps=args.denoise_num_steps,
            skip_step=args.skip_step,
            inversion_guidance=args.inversion_guidance,
            denoise_guidance=args.denoise_guidance,
            re_init=args.re_init,
            attn_mask=args.attn_mask
        )
    print(f'reinit: {opts.re_init},attn_mask: {opts.attn_mask}')

    model = Flux_kv_edit(args.device,args.name)

    json_path = os.path.join(args.dataset_path, 'mapping_file.json')
    dataset = PIE_features_Dataset(json_path=json_path,
                                    dataset_path=args.dataset_path,
                                    features_name=args.features_name,
                                    device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    print(f'rank {rank} start inference')

    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader):
            inp, inp_target, mask, output_path = data
            output_path = os.path.join(output_dir, output_path)
            if os.path.exists(output_path):
                continue
            x = model(inp, inp_target, mask, opts)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(x, output_path)
            del x, inp, inp_target, mask
            gc.collect()
            torch.cuda.empty_cache()

    print(f'rank {rank} inference done')

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Diffusion Model Inference")
    
    parser.add_argument('--inversion_num_steps', type=int, default=28, help='inversion_num_steps')
    parser.add_argument('--denoise_num_steps', type=int, default=28, help='denoise_num_steps')
    parser.add_argument('--skip_step', type=int, default=4, help='skip_step')
    
    parser.add_argument('--inversion_guidance', type=float, default=1.5, help='inversion_guidance')
    parser.add_argument('--denoise_guidance', type=float, default=6.5, help='denoise_guidance')
    parser.add_argument('--re_init', action="store_true", help='use re_init')
    parser.add_argument('--attn_mask', action="store_true", help='use attn_mask')
    
    parser.add_argument('--height', type=int, default=512, help='height')
    parser.add_argument('--width', type=int, default=512, help='width')
    
    parser.add_argument("--name", type=str, default="flux-dev", help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    parser.add_argument("--exp_name", type=str,default='debug')
    parser.add_argument("--output_dir", type=str,default='pt_result/')
    parser.add_argument("--dataset_path", type=str,default='data/pie_bench/')
    parser.add_argument("--features_name", type=str, default='none_target_prompt_features')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)