import torch
from models.kv_edit import Flux_kv_edit, SamplingOptions
from dataset.pie_bench.pie_features import PIE_features_Dataset

mode = "mem_efficient"

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

print(f"Testing: {mode}")
device = torch.device('cuda:0')
dataset = PIE_features_Dataset(
    json_path='data/pie_bench/mapping_file.json',
    dataset_path='data/pie_bench/',
    features_name='source_prompt_target_prompt_features',
    device=device)

inp, inp_target, mask, path = dataset[0]
model = Flux_kv_edit(device, 'flux-dev')
opts = SamplingOptions(width=512, height=512, inversion_num_steps=28,
                       denoise_num_steps=28, skip_step=4,
                       inversion_guidance=1.5, denoise_guidance=5.5)

import time
t0 = time.time()
x = model(inp, inp_target, mask, opts)
elapsed = time.time() - t0

print(f"nan: {x.isnan().any().item()}")
print(f"shape: {x.shape}")
print(f"min/max: {x.min().item():.3f} / {x.max().item():.3f}")
print(f"time: {elapsed:.1f}s")
