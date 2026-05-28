import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from models.rf_edit import Flux_rf_editor
from dataclasses import dataclass
from dataset.pie_bench.pie_features import PIE_features_Dataset
from tqdm import tqdm
import argparse
import os
import json

@dataclass
class SamplingOptions:
    width: int = 512
    height: int = 512
    num_steps: int = 0
    inject_step: int = 0
    guidance: float = 1.0

def custom_collate(batch):
    
    return batch[0]


def run_inference(args):
  
    dist.init_process_group('nccl',init_method='env://')
    rank = dist.get_rank()
    output_dir = os.path.join(args.output_dir, args.exp_name)
    if rank == 0:
       
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'args.json'), 'w+') as f:
            json.dump(vars(args), f, indent=4)
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    opts = SamplingOptions(
            width=args.width,
            height=args.height,
            num_steps=args.num_steps,
            inject_step=args.inject_step,
            guidance=args.guidance
        )
    
    model = Flux_rf_editor(args)

    json_path = os.path.join(args.dataset_path, 'mapping_file.json')
    dataset = PIE_features_Dataset(json_path=json_path,
                                    dataset_path=args.dataset_path,
                                    features_name='source_prompt_target_prompt_features', 
                                    device=device)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler,collate_fn=custom_collate)
    
    print(f'rank {rank} start inference')
    
    model.eval()  
    with torch.no_grad():
        for data in tqdm(dataloader):
            inp, inp_target, mask, output_path = data
            x, info = model(inp, inp_target, opts)
            del info
            output_path = os.path.join(output_dir, output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(x, output_path)
            
    print(f'rank {rank} inference done')

    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Diffusion Model Inference")
    
    parser.add_argument('--num_steps', type=int, default=25, help='num_steps')
    parser.add_argument('--inject_step', type=int, default=5, help='inject_step')
    
    parser.add_argument('--guidance', type=float, default=2, help='denoise_guidance')
    
    parser.add_argument('--height', type=int, default=512, help='height')
    parser.add_argument('--width', type=int, default=512, help='width')
    
    parser.add_argument("--name", type=str, default="flux-dev", help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    parser.add_argument("--exp_name", type=str,default='debug')
    parser.add_argument("--output_dir", type=str,default='pt_result/')
    parser.add_argument("--dataset_path", type=str,default='data/pie_bench/')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    run_inference(args) 