import os
import re
import time
from dataclasses import dataclass
from glob import iglob

from einops import rearrange,repeat
from fire import Fire
from PIL import ExifTags, Image
import cv2
from torchvision import transforms
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from flux.sampling import get_schedule, prepare, unpack
from flux.util import (configs, load_ae, load_clip, load_flow_model, load_t5)
from flux.model import Flux

@dataclass
class SamplingOptions:
    source_prompt: str = ''
    target_prompt: str = ''
    width: int = 1366
    height: int = 768
    inversion_num_steps: int = 0
    denoise_num_steps: int = 0
    inversion_guidance: float = 1.0
    denoise_guidance: float = 1.0
    seed: int | None = None


class Flux_kv_editor(torch.nn.Module):
    def __init__(self, args):
        self.device = args.device
        self.name = args.name
        super().__init__()
        self.model = load_flow_model(self.name, device=self.device,flux_cls=Flux)
     
class reconstructor_skip(Flux_kv_editor):
    def __init__(self, args):
        super().__init__(args)

    def denoise_twice_order(
    self,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float], 
    guidance: float = 4.0
):

        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)


        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            pred = self.model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            img_mid = img + (t_prev - t_curr) / 2 * pred

            t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
    
            pred_mid = self.model(
                img=img_mid,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_mid,
                guidance=guidance_vec
            )

            first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
            img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order
            
        return img

    def denoise_first_order(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        vec: Tensor,
        timesteps: list[float], 
        guidance: float = 4.0
    ):

        guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)


        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            pred = self.model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            img = img + (t_prev - t_curr) * pred
            
        return img

    @torch.inference_mode()
    def forward(self,inp,opts,mask,use_twice_order=False,inp_target=None):
        
        denoise_timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
        denoise_timesteps = denoise_timesteps[opts.skip_step:]
        h = opts.height // 8
        w = opts.width // 8
        mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=False)
        mask[mask > 0] = 1
        
        mask = repeat(mask, 'b c h w -> b (repeat c) h w', repeat=16)
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        bool_mask = (mask.sum(dim=2) > 0.5)
        mask_indices = torch.nonzero(bool_mask)[:,1]
        img_source = inp["img"].clone()
        
        noise = torch.randn_like(img_source)
        t  = denoise_timesteps[0]
        zt = img_source *(1 - t) + noise * t
        
        with torch.no_grad():
            if use_twice_order:
                inversion_timesteps = denoise_timesteps[::-1]
                z = self.denoise_twice_order(**inp, timesteps=inversion_timesteps, guidance=opts.guidance)
                inp["img"] = z
                z = self.denoise_twice_order(**inp, timesteps=denoise_timesteps, guidance=opts.guidance)
                
            else:
                inversion_timesteps = denoise_timesteps[::-1]
                z = self.denoise_first_order(**inp, timesteps=inversion_timesteps, guidance=opts.guidance)
                z = z * (1 - mask) + zt * mask
                if inp_target is not None:
                    inp_target["img"] = z
                    z = self.denoise_first_order(**inp_target, timesteps=denoise_timesteps, guidance=opts.guidance)
                else:
                    inp["img"] = z
                    z = self.denoise_first_order(**inp, timesteps=denoise_timesteps, guidance=opts.guidance)
            
            loss_mse = F.mse_loss(z * (1 - mask), img_source * (1 - mask))
            
            z = unpack(z.float(),  opts.height, opts.width)
            return z, loss_mse.item()
            


        