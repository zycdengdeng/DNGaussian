#!/usr/bin/env python
"""
Generate dense monocular depth maps for car_road dataset using Depth Anything V2.

Usage:
  python scripts/generate_depth_car_road.py \
      --images_dir data/car_road/scene1/images \
      --output_dir data/car_road/scene1/depth_maps

Requirements:
  pip install torch torchvision transformers Pillow opencv-python
"""

import argparse
import os
import numpy as np
import cv2
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Generate depth maps using Depth Anything V2')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to undistorted images directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output depth_maps directory')
    parser.add_argument('--model', type=str, default='depth-anything/Depth-Anything-V2-Small-hf',
                        help='HuggingFace model name for depth estimation')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from transformers import pipeline

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")

    pipe = pipeline(task="depth-estimation", model=args.model, device=device)

    img_files = sorted([f for f in os.listdir(args.images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    print(f"Processing {len(img_files)} images...")

    for img_file in img_files:
        img_path = os.path.join(args.images_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # Run depth estimation
        result = pipe(img)
        depth = result['depth']  # PIL Image

        # Convert to numpy
        depth_np = np.array(depth, dtype=np.float32)

        # Normalize to [0, 255]: higher value = closer (disparity convention)
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max > d_min:
            depth_norm = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_np)

        # Resize to match original image if needed
        if depth_norm.shape != (orig_h, orig_w):
            depth_norm = cv2.resize(depth_norm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        depth_png = (depth_norm * 255).astype(np.uint8)

        stem = os.path.splitext(img_file)[0]
        out_path = os.path.join(args.output_dir, f'depth_{stem}.png')
        cv2.imwrite(out_path, depth_png)
        print(f"  {img_file} -> depth_{stem}.png  (range: {d_min:.2f}-{d_max:.2f})")

    print(f"\nDone! Generated {len(img_files)} depth maps in {args.output_dir}")


if __name__ == '__main__':
    main()
