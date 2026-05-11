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

from flux.sampling import get_schedule, prepare, unpack,denoise_twice_order
from flux.util import (configs, load_ae, load_clip, load_flow_model, load_t5)
from flux.model import Flux_rf

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

class Flux_rf_editor(torch.nn.Module):
    def __init__(self, args):
        self.device = args.device
        self.name = args.name
        super().__init__()
        self.model = load_flow_model(self.name, device=self.device,flux_cls=Flux_rf)
     
    @torch.inference_mode()
    def forward(self,inp,inp_target,opts):
        info = {}
        info['feature'] = {}
        info['inject_step'] = opts.inject_step
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=True)
        
        z, info = denoise_twice_order(self.model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
        
        inp_target["img"] = z

        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=True)

        x, _ = denoise_twice_order(self.model, **inp_target, timesteps=timesteps, guidance=opts.guidance, inverse=False, info=info)

        z0 = unpack(x.float(),  opts.height, opts.width)
        return z0,info
