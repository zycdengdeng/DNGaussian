#!/usr/bin/env python
"""
Generate dense monocular depth maps for car_road dataset.

Supports multiple backends (in priority order):
  1. Local MiDaS repo (--midas_path, fully offline)
  2. torch.hub (requires GitHub access)
  3. transformers pipeline + HuggingFace mirror (for China servers)

Usage:
  # Option A: Local MiDaS (recommended if offline)
  git clone https://ghfast.top/https://github.com/isl-org/MiDaS.git third_party/MiDaS
  # Download weights: dpt_large_384.pt to third_party/MiDaS/weights/
  python scripts/generate_depth_car_road.py \
      --images_dir data/car_road/scene1/images \
      --output_dir data/car_road/scene1/depth_maps \
      --midas_path third_party/MiDaS

  # Option B: HuggingFace mirror (China)
  python scripts/generate_depth_car_road.py \
      --images_dir data/car_road/scene1/images \
      --output_dir data/car_road/scene1/depth_maps \
      --hf_mirror

  # Option C: torch.hub (needs GitHub access)
  python scripts/generate_depth_car_road.py \
      --images_dir data/car_road/scene1/images \
      --output_dir data/car_road/scene1/depth_maps

Requirements:
  pip install torch torchvision timm opencv-python Pillow
"""

import argparse
import os
import sys
import numpy as np
import cv2
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Generate depth maps for DNGaussian')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Path to undistorted images directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output depth_maps directory')
    parser.add_argument('--midas_path', type=str, default=None,
                        help='Path to local MiDaS repo (for offline use)')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='Path to MiDaS weights file (dpt_large_384.pt)')
    parser.add_argument('--model_type', type=str, default='DPT_Large',
                        choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'],
                        help='MiDaS model type')
    parser.add_argument('--hf_mirror', action='store_true', default=False,
                        help='Use hf-mirror.com (HuggingFace China mirror)')
    return parser.parse_args()


def load_midas_local(midas_path, model_type, weights_path, device):
    """Load MiDaS from local cloned repo."""
    print(f"  Loading MiDaS from local path: {midas_path}")
    sys.path.insert(0, midas_path)

    model = torch.hub.load(midas_path, model_type, source='local',
                           trust_repo=True)

    # If weights file specified, load it explicitly
    if weights_path and os.path.isfile(weights_path):
        print(f"  Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load(midas_path, "transforms", source='local',
                                       trust_repo=True)
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return model, transform


def load_midas_hub(model_type, device):
    """Load MiDaS via torch.hub (needs GitHub access)."""
    print(f"  Loading MiDaS {model_type} via torch.hub...")
    model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    model.to(device)
    model.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms",
                                       trust_repo=True)
    if model_type in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return model, transform


def load_hf_pipeline(device):
    """Load depth estimation via HuggingFace transformers (with mirror support)."""
    from transformers import DPTForDepthEstimation, DPTFeatureExtractor

    model_name = "Intel/dpt-large"
    print(f"  Loading {model_name} via transformers...")

    feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
    model = DPTForDepthEstimation.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return model, feature_extractor


def predict_depth_midas(model, transform, img_rgb, device):
    """Run MiDaS inference on a single RGB image."""
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()


def predict_depth_hf(model, feature_extractor, img_rgb, device):
    """Run HuggingFace DPT inference on a single RGB image."""
    from PIL import Image as PILImage
    orig_h, orig_w = img_rgb.shape[:2]

    inputs = feature_extractor(images=PILImage.fromarray(img_rgb), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(orig_h, orig_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    return prediction.cpu().numpy()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Set HuggingFace mirror if requested
    if args.hf_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("  Using HuggingFace mirror: https://hf-mirror.com")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Try loading model in priority order
    model = None
    transform = None
    feature_extractor = None
    use_hf = False

    # Priority 1: Local MiDaS repo
    if args.midas_path and os.path.isdir(args.midas_path):
        try:
            model, transform = load_midas_local(
                args.midas_path, args.model_type, args.weights_path, device)
            print("  Loaded MiDaS from local repo")
        except Exception as e:
            print(f"  Failed to load from local repo: {e}")

    # Priority 2: torch.hub
    if model is None and not args.hf_mirror:
        try:
            model, transform = load_midas_hub(args.model_type, device)
            print("  Loaded MiDaS via torch.hub")
        except Exception as e:
            print(f"  Failed to load via torch.hub: {e}")

    # Priority 3: HuggingFace transformers
    if model is None:
        try:
            model, feature_extractor = load_hf_pipeline(device)
            use_hf = True
            print("  Loaded DPT via transformers")
        except Exception as e:
            print(f"  Failed to load via transformers: {e}")
            print("\n  === All loading methods failed. Please try one of: ===")
            print(f"  1. Clone MiDaS locally:")
            print(f"     git clone https://ghfast.top/https://github.com/isl-org/MiDaS.git third_party/MiDaS")
            print(f"     wget -P third_party/MiDaS/weights/ https://ghfast.top/https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_large_384.pt")
            print(f"     python scripts/generate_depth_car_road.py --midas_path third_party/MiDaS ...")
            print(f"  2. pip install transformers==4.37.2 && export HF_ENDPOINT=https://hf-mirror.com")
            print(f"     python scripts/generate_depth_car_road.py --hf_mirror ...")
            sys.exit(1)

    # Process images
    img_files = sorted([f for f in os.listdir(args.images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"\nProcessing {len(img_files)} images...")

    for img_file in img_files:
        img_path = os.path.join(args.images_dir, img_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run depth estimation
        if use_hf:
            depth_np = predict_depth_hf(model, feature_extractor, img_rgb, device)
        else:
            depth_np = predict_depth_midas(model, transform, img_rgb, device)

        # Normalize to [0, 255]: higher value = closer (disparity convention)
        # MiDaS/DPT output inverse depth by default (higher = closer)
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
