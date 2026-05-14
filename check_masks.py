import json, numpy as np, os
from PIL import Image, ImageDraw, ImageFont
from models.flux_key.step2_features import load_mask_from_json
from models.flux_key.step1_extract import mask_bbox

with open('data/pie_bench/mapping_file.json') as f:
    data = json.load(f)

keys = ['000000000000','000000000001','000000000002','000000000003','000000000004']
os.makedirs('test_output/flux_key/mask_check', exist_ok=True)

for key in keys:
    item = data[key]
    src_path = 'data/pie_bench/annotation_images/' + item['image_path']
    src = Image.open(src_path).convert('RGB').resize((512,512))
    mask = load_mask_from_json('data/pie_bench/mapping_file.json', key)
    bbox = mask_bbox(mask)
    coverage = float(mask.sum()) / mask.size * 100

    overlay = Image.new('RGBA', src.size, (0,0,0,0))
    mask_img = Image.fromarray((mask*128).astype('uint8'), 'L')
    red = Image.new('RGBA', src.size, (255,0,0,128))
    overlay.paste(red, mask=mask_img)
    vis = Image.alpha_composite(src.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(vis)
    draw.rectangle(bbox, outline='yellow', width=3)

    op = key
    ep = item['editing_prompt'][:50]
    print(f"{key}: bbox={bbox}  coverage={coverage:.1f}%")
    print(f"  orig:  {item['original_prompt'][:60]}")
    print(f"  edit:  {ep}")
    print(f"  mask_tokens_approx: {int(mask.sum()/4)}")
    print()

    vis.save(f'test_output/flux_key/mask_check/{key}.jpg', quality=92)

print("Mask visualizations saved to test_output/flux_key/mask_check/")
