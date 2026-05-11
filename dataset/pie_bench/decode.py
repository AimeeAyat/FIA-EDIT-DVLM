import os
from pathlib import Path
import torch
from PIL import Image
from dataset.pie_bench.pie_features import pie_features_extractor
from einops import rearrange
from flux.sampling import unpack
from tqdm import tqdm
import argparse
def process_pt_file(pt_file_path):
    data = torch.load(pt_file_path)
    image = ...
    return image

def save_image(image, output_image_path):
    output_image_dir = os.path.dirname(output_image_path)
    os.makedirs(output_image_dir, exist_ok=True)
    image.save(output_image_path)

def decode_pt_file(dir_path, output_path,device):
    input_dir = dir_path
    if input_dir.endswith('/'):
        input_dir = input_dir[:-1]
    input_root = os.path.dirname(input_dir)
    output_root = output_path
    device = torch.device(device)
    extractor = pie_features_extractor(device)
    decode_list = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pt'):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_root)
                output_subdir = os.path.join(output_root, relative_path)
                output_image_path = os.path.join(output_subdir, file.replace('.pt', '.jpg'))
                os.makedirs(output_subdir, exist_ok=True)
                decode_list.append((input_file_path, output_image_path))
                
    for input_file_path, output_image_path in tqdm(decode_list):
        x = torch.load(input_file_path,weights_only=True,map_location=device)
        with torch.autocast(device_type=extractor.device.type, dtype=torch.bfloat16):
            x = extractor.ae.decode(x.to(extractor.device))
        x = x.clamp(-1, 1)
        x = x.cpu()
        x = rearrange(x[0], "c h w -> h w c")
        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        img.save(output_image_path, quality=95, subsampling=0)
def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Diffusion Model Inference")
    
    parser.add_argument("--dir_path", type=str, default="pt_result/step_28_skip_4_1.5_5.5_kv_edit")
    parser.add_argument("--output_path", type=str, default="output/")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    decode_pt_file(args.dir_path, args.output_path, 'cuda')