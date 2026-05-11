"""
Feature extraction with configurable source prompt and features name.

Usage:
  # Paper default (empty source prompt):
  python extract_features.py --features_name none_target_prompt_features --empty_source

  # Source prompt variant:
  python extract_features.py --features_name source_prompt_target_prompt_features
"""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from dataset.pie_bench.pie_features import PIE_Dataset, pie_features_extractor, custom_collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path",       default="data/pie_bench/mapping_file.json")
    parser.add_argument("--dataset_path",    default="data/pie_bench/")
    parser.add_argument("--features_name",   default="none_target_prompt_features",
                        help="Subfolder name for saved features")
    parser.add_argument("--empty_source",    action="store_true",
                        help="Use empty string as source prompt (paper default)")
    parser.add_argument("--num_workers",     type=int, default=4)
    parser.add_argument("--device",          default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = PIE_Dataset(
        json_path=args.json_path,
        dataset_path=args.dataset_path,
        features_name=args.features_name,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
    )

    extractor = pie_features_extractor(device=args.device)

    print(f"Extracting features → {args.features_name}  (empty_source={args.empty_source})")

    for batch in tqdm(dataloader):
        images, masks, source_prompt, target_prompt, output_path = batch

        # skip if already extracted
        if os.path.exists(output_path[0]):
            continue

        if args.empty_source:
            source_prompt = [""]

        inp, inp_target = extractor.extract_features(images, source_prompt, target_prompt)
        os.makedirs(os.path.dirname(output_path[0]), exist_ok=True)
        torch.save({"inp": inp, "inp_target": inp_target, "mask": masks}, output_path[0])


if __name__ == "__main__":
    main()
