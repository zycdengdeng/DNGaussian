#!/usr/bin/env python
"""
Convert Depth Anything V3 metric depth (.npy) to DNGaussian format (8-bit PNG).

DA3 output:  float32 metric depth, farther = higher value
DNGaussian:  8-bit PNG, closer = brighter (disparity-style)
             Training does `255.0 - depth_mono` to flip convention.

Usage:
  python scripts/convert_da3_depth.py \
      --depth_dir data/car_road/scene1/depth \
      --images_dir data/car_road/scene1/images \
      --output_dir data/car_road/scene1/depth_maps
"""

import argparse
import os
import numpy as np
import cv2


def main():
    parser = argparse.ArgumentParser(description='Convert DA3 depth to DNGaussian format')
    parser.add_argument('--depth_dir', type=str, required=True,
                        help='Directory with DA3 *_depth.npy files')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory with image files (for name matching)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output depth_maps directory')
    parser.add_argument('--max_depth', type=float, default=None,
                        help='Clip max depth (default: use per-image max)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build mapping: pinholeX prefix -> image filename
    img_files = sorted([f for f in os.listdir(args.images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    npy_files = sorted([f for f in os.listdir(args.depth_dir) if f.endswith('_depth.npy')])

    # Extract prefix (e.g. "pinhole0") from npy filenames
    prefix_to_img = {}
    for img_f in img_files:
        for npy_f in npy_files:
            prefix = npy_f.replace('_depth.npy', '')  # e.g. "pinhole0"
            if img_f.startswith(prefix):
                prefix_to_img[prefix] = img_f

    print(f"Found {len(npy_files)} depth maps, {len(img_files)} images")
    print(f"Matched {len(prefix_to_img)} pairs:\n")

    for npy_f in npy_files:
        prefix = npy_f.replace('_depth.npy', '')
        if prefix not in prefix_to_img:
            print(f"  WARNING: no image match for {npy_f}, skipping")
            continue

        img_name = prefix_to_img[prefix]
        img_stem = os.path.splitext(img_name)[0]

        # Load metric depth (farther = higher)
        depth = np.load(os.path.join(args.depth_dir, npy_f)).astype(np.float32)

        # Clip minimum to avoid division by zero
        depth = np.clip(depth, 1e-3, args.max_depth if args.max_depth else depth.max())

        # Convert to inverse depth (disparity): closer = higher
        inv_depth = 1.0 / depth

        # Normalize to [0, 255]
        d_min, d_max = inv_depth.min(), inv_depth.max()
        if d_max > d_min:
            depth_norm = (inv_depth - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(inv_depth)

        depth_u8 = (depth_norm * 255.0).astype(np.uint8)

        # Save as depth_<image_stem>.png
        out_path = os.path.join(args.output_dir, f'depth_{img_stem}.png')
        cv2.imwrite(out_path, depth_u8)

        print(f"  {npy_f} -> depth_{img_stem}.png")
        print(f"    metric depth: [{depth.min():.3f}, {depth.max():.3f}]")
        print(f"    inv depth:    [{inv_depth.min():.4f}, {inv_depth.max():.4f}]")
        print(f"    output range: [{depth_u8.min()}, {depth_u8.max()}]")

    print(f"\nDone! Output in {args.output_dir}")


if __name__ == '__main__':
    main()
