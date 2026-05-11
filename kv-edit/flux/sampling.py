import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux, Flux_fe,Flux_rf,Flux_kv
from .modules.conditioner import HFEmbedder
from tqdm import tqdm

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    

def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

def prepare_flowedit(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, source_prompt: str | list[str],target_prompt) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(source_prompt, str):
        bs = len(source_prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)


    if isinstance(source_prompt, str):
        source_prompt = [source_prompt]
    source_txt = t5(source_prompt)
    if source_txt.shape[0] == 1 and bs > 1:
        source_txt = repeat(source_txt, "1 ... -> bs ...", bs=bs)
    source_txt_ids = torch.zeros(bs, source_txt.shape[1], 3)
    
    source_vec = clip(target_prompt)
    if source_vec.shape[0] == 1 and bs > 1:
        source_vec = repeat(source_vec, "1 ... -> bs ...", bs=bs)
    
    if isinstance(target_prompt, str):
        target_prompt = [target_prompt]
    target_txt = t5(target_prompt)
    if target_txt.shape[0] == 1 and bs > 1:
        target_txt = repeat(target_txt, "1 ... -> bs ...", bs=bs)
    target_txt_ids = torch.zeros(bs, target_txt.shape[1], 3)
    
    target_vec = clip(target_prompt)
    if target_vec.shape[0] == 1 and bs > 1:
        target_vec = repeat(target_vec, "1 ... -> bs ...", bs=bs)
    

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "source_txt": source_txt.to(img.device),
        "source_txt_ids": source_txt_ids.to(img.device),
        "source_vec": source_vec.to(img.device),
        "target_txt": target_txt.to(img.device),
        "target_txt_ids": target_txt_ids.to(img.device),
        "target_vec": target_vec.to(img.device)
    }

def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)

    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
):
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
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

def denoise_flow_edit(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    source_txt: Tensor,
    source_txt_ids: Tensor,
    source_vec: Tensor,
    target_txt: Tensor,
    target_txt_ids: Tensor,
    target_vec: Tensor,
    timesteps: list[float],
    target_guidance: float = 4.0,
    source_guidance: float = 4.0
):
    init_img = img.clone()
    z_fe = img.clone()
    target_guidance_vec = torch.full((img.shape[0],), target_guidance, device=img.device, dtype=img.dtype)
    source_guidance_vec = torch.full((img.shape[0],), source_guidance, device=img.device, dtype=img.dtype)
    
    noise_list = []
    for i in range(len(timesteps)):
        noise = torch.randn(init_img.size(), dtype=init_img.dtype, 
                        layout=init_img.layout, device=init_img.device,
                        generator=torch.Generator(device=init_img.device).manual_seed(0))
        noise_list.append(noise)
    
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
    
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        
        z_src = (1 - t_curr) * init_img + t_curr * noise_list[i]
        z_tar =  z_src - init_img + z_fe
        
        v_tar = model(
            img=z_tar,
            img_ids=img_ids,
            txt=target_txt,
            txt_ids=target_txt_ids,
            y=target_vec,
            timesteps=t_vec,
            guidance=target_guidance_vec
        )
        
        v_src = model(
            img=z_src,
            img_ids=img_ids,
            txt=source_txt,
            txt_ids=source_txt_ids,
            y=source_vec,
            timesteps=t_vec,
            guidance=source_guidance_vec
        )
        
        v_fe = v_tar - v_src
        z_fe = z_fe + (t_prev - t_curr) * v_fe
    return z_fe

def denoise_twice_order(
    model: Flux_rf,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

    return img, info

def denoise_frist_order(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    step_vec_list = []
    info[f'{"inversion" if inverse else "denoise"}_step_img'] = []
    info['qkp'] = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]
        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        img = img + (t_prev - t_curr) * pred
        step_vec_list.append(pred.cpu())
        info[f'{"inversion" if inverse else "denoise"}_step_img'].append(img)
        
    torch.save(info['qkp'], f'feature/attention_map/{"inversion" if inverse else "denoise"}_qkp.pt')
    return img, info

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )

def denoise_kv(
    model: Flux_kv,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):

    if inverse:
        timesteps = timesteps[::-1]
        
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        
        if inverse:
            img_name = str(info['t']) + '_' + 'img'
            info['feature'][img_name] = img.cpu()
        else:
            img_name = str(info['t']) + '_' + 'img'
            source_img = info['feature'][img_name].to(img.device)
            img = source_img[:, info['mask_indices'],...] * (1 - info['mask'][:, info['mask_indices'],...]) + img * info['mask'][:, info['mask_indices'],...]
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )
    
        img = img + (t_prev - t_curr) * pred
    return img, info
def denoise_kv_twice_order(
    model: Flux_kv,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):

    if inverse:
        timesteps = timesteps[::-1]
        
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        
        if inverse:
            img_name = str(info['t']) + '_' + 'img'
            info['feature'][img_name] = img.cpu()
        else:
            img_name = str(info['t']) + '_' + 'img'
            source_img = info['feature'][img_name].to(img.device)
            img = source_img[:, info['mask_indices'],...] * (1 - info['mask'][:, info['mask_indices'],...]) + img * info['mask'][:, info['mask_indices'],...]
            
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )
        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['t'] = t_curr + (t_prev - t_curr) / 2
        pred_mid = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order
        

    return img, info

def denoise_kv_inf(
    model: Flux_kv,
    img: Tensor,
    img_ids: Tensor,
    source_txt: Tensor,
    source_txt_ids: Tensor,
    source_vec: Tensor,
    target_txt: Tensor,
    target_txt_ids: Tensor,
    target_vec: Tensor,
    timesteps: list[float],
    target_guidance: float = 4.0,
    source_guidance: float = 4.0,
    info: dict = {},
):
        
    target_guidance_vec = torch.full((img.shape[0],), target_guidance, device=img.device, dtype=img.dtype)
    source_guidance_vec = torch.full((img.shape[0],), source_guidance, device=img.device, dtype=img.dtype)
    
    mask_indices = info['mask_indices']
    init_img = img.clone()
    z_fe = img[:, mask_indices,...]
    
    noise_list = []
    for i in range(len(timesteps)):
        noise = torch.randn(init_img.size(), dtype=init_img.dtype, 
                        layout=init_img.layout, device=init_img.device,
                        generator=torch.Generator(device=init_img.device).manual_seed(0))
        noise_list.append(noise)
    mid = None
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        
        info['t'] = t_curr
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        
        z_src = (1 - t_curr) * init_img + t_curr * noise_list[i]
        z_tar = z_src[:, mask_indices,...] - init_img[:, mask_indices,...] + z_fe
        
        info['inverse'] = True
        info['feature'] = {}
        v_src = model(
            img=z_src,
            img_ids=img_ids,
            txt=source_txt,
            txt_ids=source_txt_ids,
            y=source_vec,
            timesteps=t_vec,
            guidance=source_guidance_vec,
            info=info
        )
        
        info['inverse'] = False
        v_tar = model(
            img=z_tar,
            img_ids=img_ids,
            txt=target_txt,
            txt_ids=target_txt_ids,
            y=target_vec,
            timesteps=t_vec,
            guidance=target_guidance_vec,
            info=info
        )
    
        
        v_fe = v_tar - v_src[:, mask_indices,...]
        z_fe = z_fe + (t_prev - t_curr) * v_fe * info['mask'][:, mask_indices,...]
        if i == 12:
            mid = z_fe.clone()
    return mid, info
