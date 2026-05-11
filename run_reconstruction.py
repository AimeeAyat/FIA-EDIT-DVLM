import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from models.reconstruction import reconstructor_skip
from dataclasses import dataclass
from dataset.pie_bench.pie_features import PIE_features_Dataset
from tqdm import tqdm
import argparse
import os
import json
import csv

@dataclass
class SamplingOptions:
    # prompt: str
    width: int = 512
    height: int = 512
    num_steps: int = 0
    skip_step: int = 0
    guidance: float = 1.0
    
def custom_collate(batch):
    
    return batch[0]

def run_inference(args):

    dist.init_process_group('nccl',init_method='env://')

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
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
            skip_step=args.skip_step,
            guidance=args.guidance,
        )

    model = reconstructor_skip(args)
    
    json_path = os.path.join(args.dataset_path, 'mapping_file.json')
    dataset = PIE_features_Dataset(json_path=json_path,
                                    dataset_path=args.dataset_path,
                                    features_name='source_prompt_target_prompt_features', 
                                    device=device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,collate_fn=custom_collate)
    
    print(f'rank {rank} start inference')
    model.eval()  
    loss_sum = 0.0
    with torch.no_grad():  
        for data in tqdm(dataloader):
            inp, inp_target, mask, output_path = data
            z, loss_mse = model(inp,opts,mask,use_twice_order=False,inp_target=inp_target)
            
            with open(os.path.join(output_dir,'mse.csv'),'a+',newline="") as f:
                csv_write = csv.writer(f)
                csv_write.writerow([output_path,loss_mse])
                
            loss_sum += loss_mse
            output_path = os.path.join(output_dir, output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save(z, output_path)

            
    loss = loss_sum / len(dataloader)
    print(f'experiment {args.exp_name} mse loss: {loss}')
    
    with open(os.path.join(output_dir,'mse.csv'),'a+',newline="") as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['loss_ave',loss])
    
    print(f'rank {rank} inference done')

    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Diffusion Model Inference")
    
    parser.add_argument('--num_steps', type=int, default=28, help='inversion_num_steps')
    parser.add_argument('--skip_step', type=int, default=0, help='skip_step')
    
    parser.add_argument('--guidance', type=float, default=1.5, help='inversion_guidance')
    
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