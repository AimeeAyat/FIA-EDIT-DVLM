import os
import sys

import time
from io import BytesIO
import uuid
from dataclasses import dataclass
from glob import iglob
import argparse
from einops import rearrange
from fire import Fire
from PIL import ExifTags, Image
from tqdm import tqdm
from torchvision import transforms
import torch
import torch.nn.functional as F
import json
import numpy as np
from transformers import pipeline
from torch.utils.data import Dataset,DataLoader
from flux.sampling import prepare
from flux.util import (configs, embed_watermark, load_ae, load_clip, load_t5)

def custom_collate_fn(batch):

    images, masks, source_prompts, target_prompts, output_paths = zip(*batch)
    images_stacked = torch.cat(images, dim=0)
    masks_stacked = torch.cat(masks, dim=0)
    strings1_list = list(source_prompts)
    strings2_list = list(target_prompts)
    strings3_list = list(output_paths)
    
    return images_stacked, masks_stacked, strings1_list, strings2_list, strings3_list

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))

    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    mask_array[0,:]=0
    mask_array[-1,:]=0
    mask_array[:,0]=0
    mask_array[:,-1]=0
            
    return mask_array
class PIE_Dataset(Dataset):
    def __init__(self, json_path, dataset_path, features_name):
        super().__init__()
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.keys = list(self.data.keys())
        self.values = list(self.data.values())
        self.features_path = os.path.join(dataset_path, features_name)
        os.makedirs(self.features_path, exist_ok=True)
        self.image_dir =  os.path.join(dataset_path, "annotation_images")
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.values[idx]

        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{self.image_dir}", item["image_path"])
        output_path = os.path.join(f"{self.features_path}", item["image_path"].replace(".jpg", ".pt"))
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1
        image = image.unsqueeze(0)
        
        mask = np.uint8(mask_decode(item["mask"]))
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
        return image, mask, original_prompt, editing_prompt, output_path

class PIE_features_Dataset(PIE_Dataset):
    def __init__(self, json_path, dataset_path, features_name,device):
        super().__init__(json_path, dataset_path, features_name)
        self.device = device    
        
    def __getitem__(self, idx):
        item = self.values[idx]
        pt_path = os.path.join(f"{self.features_path}", item["image_path"].replace(".jpg", ".pt"))
        data = torch.load(pt_path,weights_only=True, map_location=self.device)
        inp, inp_target, mask = data['inp'], data['inp_target'], data['mask']
        
        return inp, inp_target, mask, item["image_path"].replace(".jpg", ".pt")

    def __len__(self):
        return len(self.data)
        # return 3
    

class pie_features_extractor():
    def __init__(self,device):
        self.device = device
        self.t5 = load_t5(self.device, max_length=512)
        self.clip = load_clip(self.device)
        self.ae = load_ae("flux-dev", self.device)
    
    def extract_features(self, init_img, source_prompt, target_prompt):
        
        init_img = self.ae.encode(init_img.to(self.device)).to(torch.bfloat16)
        with torch.no_grad():
            inp = prepare(self.t5, self.clip, init_img, prompt=source_prompt)
            inp_target = prepare(self.t5, self.clip, init_img, prompt=target_prompt)
        
        return inp, inp_target
        
        
if __name__ == "__main__":
    dataset = PIE_Dataset(
    json_path='data/pie_bench/mapping_file.json',
    dataset_path='data/pie_bench/',
    features_name='source_prompt_target_prompt_features'
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
    )   
    
    extractor = pie_features_extractor(device='cuda')
    
    for batch in tqdm(dataloader):
        images, masks, source_prompt, target_prompt, output_path = batch
        # source_prompt = [""]
        inp, inp_target = extractor.extract_features(images, source_prompt, target_prompt)
        
        os.makedirs(os.path.dirname(output_path[0]), exist_ok=True)
        torch.save({'inp': inp, 'inp_target': inp_target, 'mask': masks}, output_path[0])