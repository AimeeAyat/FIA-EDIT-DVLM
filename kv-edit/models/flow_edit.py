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

from flux.sampling import get_schedule, prepare, unpack,denoise_flow_edit
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

class Flux_fe_editor(torch.nn.Module):
    def __init__(self, args):
        self.device = args.device
        self.name = args.name
        super().__init__()
        self.model = load_flow_model(self.name, device=self.device,flux_cls=Flux)
     
    @torch.inference_mode()
    def forward(self,inp,inp_target,opts):
        denoise_timesteps = get_schedule(opts.denoise_num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
        denoise_timesteps = denoise_timesteps[opts.skip_step:]
        
        z = denoise_flow_edit(self.model, img=inp["img"], img_ids=inp['img_ids'], 
                                    source_txt=inp['txt'], source_txt_ids=inp['txt_ids'], source_vec=inp['vec'],
                                    target_txt=inp_target['txt'], target_txt_ids=inp_target['txt_ids'], target_vec=inp_target['vec'],
                                    timesteps=denoise_timesteps, source_guidance=opts.inversion_guidance, target_guidance=opts.denoise_guidance)
        
        z0 = unpack(z.float(),  opts.height, opts.width)
        return z0
