#!/usr/bin/env python
"""
Generate dense monocular depth maps for car_road dataset using MiDaS DPT-Large.
Uses torch.hub (compatible with PyTorch >= 2.0, no transformers dependency).

Usage:
  python scripts/generate_depth_car_road.py \
      --images_dir data/car_road/scene1/images \
      --output_dir data/car_road/scene1/depth_maps

Requirements:
  pip install torch torchvision timm Pillow opencv-python
"""

import argparse
import os
import numpy as np
import cv2
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Generate depth maps using MiDaS DPT')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to undistorted images directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output depth_maps directory')
    parser.add_argument('--model_type', type=str, default='DPT_Large',
                        choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
                        help='MiDaS model type (default: DPT_Large)')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load MiDaS model via torch.hub (works with PyTorch 2.x)
    print(f"Loading MiDaS {args.model_type}...")
    model = torch.hub.load("intel-isl/MiDaS", args.model_type, trust_repo=True)
    model.to(device)
    model.eval()

    # Load MiDaS transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    if args.model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    img_files = sorted([f for f in os.listdir(args.images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Processing {len(img_files)} images...")

    for img_file in img_files:
        img_path = os.path.join(args.images_dir, img_file)

        # Read image as RGB
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        # Transform and predict
        input_batch = transform(img_rgb).to(device)

        with torch.no_grad():
            prediction = model(input_batch)
            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(orig_h, orig_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # MiDaS outputs inverse depth (disparity): higher = closer
        depth_np = prediction.cpu().numpy()

        # Normalize to [0, 255]: higher value = closer (matches DNGaussian convention)
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max > d_min:
            depth_norm = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_np)

        depth_png = (depth_norm * 255).astype(np.uint8)

        stem = os.path.splitext(img_file)[0]
        out_path = os.path.join(args.output_dir, f'depth_{stem}.png')
        cv2.imwrite(out_path, depth_png)
        print(f"  {img_file} -> depth_{stem}.png  (range: {d_min:.2f}-{d_max:.2f})")

    print(f"\nDone! Generated {len(img_files)} depth maps in {args.output_dir}")


if __name__ == '__main__':
    main()
