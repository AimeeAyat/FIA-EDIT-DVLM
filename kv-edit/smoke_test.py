import torch
from dataset.pie_bench.pie_features import pie_features_extractor
from einops import rearrange
from PIL import Image
import os

device = torch.device('cuda:0')
extractor = pie_features_extractor(device)

files = [
    'pt_result/step_28_skip_4_1.5_5.5/0_random_140/000000000000.pt',
    'pt_result/step_28_skip_4_1.5_5.5/0_random_140/000000000001.pt',
    'pt_result/step_28_skip_4_1.5_5.5/0_random_140/000000000002.pt',
]
os.makedirs('smoke_test_output', exist_ok=True)
for f in files:
    x = torch.load(f, weights_only=True, map_location=device)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        x = extractor.ae.decode(x.to(device))
    x = x.clamp(-1, 1).cpu()
    x = rearrange(x[0], 'c h w -> h w c')
    img = Image.fromarray((127.5 * (x + 1.0)).byte().numpy())
    out = 'smoke_test_output/' + os.path.basename(f).replace('.pt', '.jpg')
    img.save(out, quality=95)
    gray = img.convert('L')
    print(f"{out} | size={img.size} | pixel_range={gray.getextrema()}")
