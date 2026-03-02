#!/usr/bin/env python
"""
Prepare car_road roadside camera dataset for DNGaussian training.

Converts:
  - calib.json (camera intrinsics/extrinsics in virtualLidar frame)
  - LiDAR point cloud (.pcd)
  - Pinhole camera images

Into DNGaussian-compatible COLMAP format:
  - sparse/0/cameras.txt
  - sparse/0/images.txt
  - sparse/0/points3D.txt (empty)
  - sparse/0/points3D.ply (colored LiDAR point cloud)
  - images/ (undistorted images)
  - depth_maps/ (depth maps from LiDAR projection)

Usage:
  python scripts/prepare_car_road.py \
      --input_dir self_Dataset \
      --output_dir data/car_road/scene1

Then train:
  python train_llff.py -s data/car_road/scene1 \
      --model_path output/car_road/scene1 \
      -r 1 --eval --n_sparse 3
"""

import argparse
import json
import os
import glob
import sys
import numpy as np
import cv2
from PIL import Image


# ============================================================
# Mapping: pinhole folder name -> camera key in calib.json
# pinhole0 -> cam3, pinhole1 -> cam6, pinhole2 -> cam9, pinhole3 -> cam0
# ============================================================
PINHOLE_CAM_MAP = {
    'pinhole0': '3',
    'pinhole1': '6',
    'pinhole2': '9',
    'pinhole3': '0',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare car_road dataset for DNGaussian')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to self_Dataset folder')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output dataset folder')
    parser.add_argument('--filter_visible', action='store_true', default=True,
                        help='Only keep LiDAR points visible in at least one camera')
    parser.add_argument('--skip_depth', action='store_true', default=False,
                        help='Skip depth map generation (run separately later)')
    return parser.parse_args()


# ============================================================
# PCD reader
# ============================================================
def read_pcd_ascii(path):
    """Read ASCII PCD file. Returns (N,3) xyz and (N,) intensity arrays."""
    with open(path, 'r') as f:
        lines = f.readlines()

    data_start = 0
    num_points = 0
    for i, line in enumerate(lines):
        if line.startswith('POINTS'):
            num_points = int(line.strip().split()[-1])
        if line.startswith('DATA'):
            data_start = i + 1
            break

    points = []
    intensities = []
    for line in lines[data_start:data_start + num_points]:
        parts = line.strip().split()
        if len(parts) >= 4:
            x, y, z, intensity = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            # Filter invalid points
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                points.append([x, y, z])
                intensities.append(intensity)

    return np.array(points, dtype=np.float64), np.array(intensities, dtype=np.float64)


# ============================================================
# Coordinate conversion utilities
# ============================================================
def rodrigues_to_rotmat(rvec):
    """Convert Rodrigues rotation vector to 3x3 rotation matrix."""
    rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    return R


def rotmat2qvec(R):
    """Convert 3x3 rotation matrix to COLMAP quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[:, np.argmax(eigvals)]
    if qvec[3] < 0:
        qvec *= -1
    return np.array([qvec[3], qvec[0], qvec[1], qvec[2]])


# ============================================================
# Image processing
# ============================================================
def undistort_image_and_intrinsics(img, K, dist_coeffs):
    """
    Undistort image and compute new camera intrinsics.
    Returns: undistorted image, new camera matrix K_new
    """
    h, w = img.shape[:2]
    # alpha=0: crop to valid region, no black borders
    K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha=0, newImgSize=(w, h))
    img_undist = cv2.undistort(img, K, dist_coeffs, None, K_new)
    return img_undist, K_new


# ============================================================
# Point cloud projection and coloring
# ============================================================
def project_points(points_3d, K, R, t):
    """
    Project 3D points (world/virtualLidar frame) to 2D image coordinates.
    R, t: world-to-camera transform (P_cam = R @ P_world + t)
    Returns: (N, 2) pixel coords, (N,) depths in camera frame
    """
    pts_cam = (R @ points_3d.T).T + t.reshape(1, 3)
    depths = pts_cam[:, 2].copy()

    # Avoid division by zero
    safe_z = pts_cam[:, 2].copy()
    safe_z[safe_z == 0] = 1e-10

    pts_2d = np.zeros((len(points_3d), 2))
    pts_2d[:, 0] = K[0, 0] * pts_cam[:, 0] / safe_z + K[0, 2]
    pts_2d[:, 1] = K[1, 1] * pts_cam[:, 1] / safe_z + K[1, 2]

    return pts_2d, depths


def color_pointcloud(points_3d, cameras_list, images_dict):
    """
    Assign RGB colors to 3D points by projecting onto camera images.
    Uses the camera with the smallest depth (closest) for each point.
    Returns: (N, 3) uint8 colors, (N,) bool mask of colored points
    """
    N = points_3d.shape[0]
    colors = np.ones((N, 3), dtype=np.uint8) * 128  # default gray
    min_depth = np.full(N, np.inf)
    colored = np.zeros(N, dtype=bool)

    for cam_key, cam in cameras_list.items():
        K = cam['K_new']
        R = cam['R']
        t = cam['t']
        img = images_dict[cam_key]  # RGB numpy array
        h, w = img.shape[:2]

        pts_2d, depths = project_points(points_3d, K, R, t)

        # Valid: in front of camera, within image bounds
        valid = (depths > 0.5) & \
                (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < w - 1) & \
                (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < h - 1)

        valid_idx = np.where(valid)[0]
        u = pts_2d[valid_idx, 0].astype(int)
        v = pts_2d[valid_idx, 1].astype(int)

        # Keep color from closest camera
        closer = depths[valid_idx] < min_depth[valid_idx]
        update_idx = valid_idx[closer]

        colors[update_idx] = img[v[closer], u[closer]]
        min_depth[update_idx] = depths[valid_idx][closer]
        colored[update_idx] = True

    return colors, colored


# ============================================================
# Depth map generation from LiDAR
# ============================================================
def create_depth_map(points_3d, K, R, t, width, height):
    """
    Create sparse depth map by projecting LiDAR points onto camera.
    Returns: (H, W) float32 depth map, 0 where no data.
    """
    pts_2d, depths = project_points(points_3d, K, R, t)

    valid = (depths > 0.5) & \
            (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < width) & \
            (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < height)

    depth_map = np.zeros((height, width), dtype=np.float32)

    valid_idx = np.where(valid)[0]
    u = pts_2d[valid_idx, 0].astype(int)
    v = pts_2d[valid_idx, 1].astype(int)
    d = depths[valid_idx].astype(np.float32)

    # For multiple points at same pixel, keep closest
    for i in range(len(valid_idx)):
        if depth_map[v[i], u[i]] == 0 or d[i] < depth_map[v[i], u[i]]:
            depth_map[v[i], u[i]] = d[i]

    return depth_map


def densify_depth_map(depth_map, iterations=15, kernel_size=5):
    """Densify sparse depth map using iterative dilation."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = depth_map.copy()

    for _ in range(iterations):
        dilated = cv2.dilate(result, kernel)
        fill_mask = (result == 0) & (dilated > 0)
        result[fill_mask] = dilated[fill_mask]

    return result


def depth_to_disparity_png(depth_map, max_depth=200.0):
    """
    Convert metric depth map to 8-bit disparity-style PNG.
    DNGaussian convention: higher pixel value = closer to camera.
    In training: 255.0 - depth_mono flips it so higher = farther (matches rendered depth).
    """
    # Clip depth to valid range
    depth_clipped = np.clip(depth_map, 0.01, max_depth)

    # Convert to inverse depth (disparity)
    disparity = 1.0 / depth_clipped

    # Normalize to [0, 255]: close objects (high disparity) -> high pixel value
    disp_min = 1.0 / max_depth
    disp_max = 1.0 / 0.5  # minimum depth 0.5m
    disparity_norm = (disparity - disp_min) / (disp_max - disp_min)
    disparity_norm = np.clip(disparity_norm, 0, 1)

    # Scale to uint8
    depth_png = (disparity_norm * 255).astype(np.uint8)

    # Set invalid areas (depth=0) to 0
    depth_png[depth_map == 0] = 0

    return depth_png


# ============================================================
# PLY writer (same format as DNGaussian's storePly)
# ============================================================
def write_ply(path, xyz, rgb):
    """Write colored point cloud as PLY (compatible with fetchPly)."""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        print("Warning: plyfile not installed. Writing ASCII PLY fallback.")
        write_ply_ascii(path, xyz, rgb)
        return

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz, dtype=np.float32)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz.astype(np.float32), normals, rgb.astype(np.float32)), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(path)


def write_ply_ascii(path, xyz, rgb):
    """Fallback ASCII PLY writer."""
    N = xyz.shape[0]
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            f.write(f"{xyz[i,0]} {xyz[i,1]} {xyz[i,2]} 0 0 0 "
                    f"{int(rgb[i,0])} {int(rgb[i,1])} {int(rgb[i,2])}\n")


# ============================================================
# Dense monocular depth estimation (Depth Anything V2)
# ============================================================
def generate_depth_maps(images_dir, depth_maps_dir):
    """
    Generate dense monocular depth maps using MiDaS DPT-Large via torch.hub.
    Saves 8-bit grayscale PNGs: higher value = closer (disparity convention).
    This matches DNGaussian's expected format (inverted with 255-x in training).
    """
    try:
        import torch
    except ImportError:
        print("  ERROR: PyTorch not installed. Run depth generation separately:")
        print(f"    python scripts/generate_depth_car_road.py "
              f"--images_dir {images_dir} --output_dir {depth_maps_dir}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Loading MiDaS DPT_Large (device={device})...")

    try:
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
        model.to(device)
        model.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = midas_transforms.dpt_transform
    except Exception as e:
        print(f"  ERROR: Failed to load MiDaS: {e}")
        print(f"  Run depth generation separately with:")
        print(f"    python scripts/generate_depth_car_road.py "
              f"--images_dir {images_dir} --output_dir {depth_maps_dir}")
        return

    img_files = sorted([f for f in os.listdir(images_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for img_file in img_files:
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=(orig_h, orig_w),
                mode="bicubic", align_corners=False,
            ).squeeze()

        depth_np = prediction.cpu().numpy()
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max > d_min:
            depth_norm = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_np)

        depth_png = (depth_norm * 255).astype(np.uint8)
        stem = os.path.splitext(img_file)[0]
        out_path = os.path.join(depth_maps_dir, f'depth_{stem}.png')
        cv2.imwrite(out_path, depth_png)
        print(f"  {img_file} -> depth_{stem}.png  (range: {d_min:.2f}-{d_max:.2f})")

    print(f"  Generated {len(img_files)} depth maps")


# ============================================================
# COLMAP text file writers
# ============================================================
def write_cameras_txt(cameras_list, output_path):
    """Write COLMAP cameras.txt (PINHOLE model)."""
    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras_list)}\n")
        for cam_id, cam in cameras_list.items():
            K = cam['K_new']
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            w, h = cam['width'], cam['height']
            f.write(f"{cam['colmap_cam_id']} PINHOLE {w} {h} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")


def write_images_txt(cameras_list, output_path):
    """Write COLMAP images.txt."""
    with open(output_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(cameras_list)}\n")
        for img_id, (cam_key, cam) in enumerate(cameras_list.items(), 1):
            qvec = rotmat2qvec(cam['R'])
            t = cam['t']
            name = cam['image_name']
            cam_id = cam['colmap_cam_id']
            f.write(f"{img_id} {qvec[0]:.10f} {qvec[1]:.10f} {qvec[2]:.10f} {qvec[3]:.10f} "
                    f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {cam_id} {name}\n")
            f.write("\n")  # empty line for 2D points


# ============================================================
# Main pipeline
# ============================================================
def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    # Load calibration
    calib_path = os.path.join(input_dir, 'calib.json')
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    print(f"Loaded calibration from {calib_path}")

    # Create output directories
    os.makedirs(os.path.join(output_dir, 'sparse', '0'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'depth_maps'), exist_ok=True)

    # ============================================================
    # Step 1: Process cameras - extract intrinsics, extrinsics, undistort images
    # ============================================================
    print("\n=== Step 1: Processing cameras ===")
    cameras = {}  # key -> camera data dict
    images = {}   # key -> RGB numpy array (undistorted)

    colmap_cam_id = 0
    for pinhole_id in sorted(PINHOLE_CAM_MAP.keys()):
        cam_key = PINHOLE_CAM_MAP[pinhole_id]
        cam_calib = calib['camera'][cam_key]

        if cam_calib['isFish'] != 0:
            print(f"  Skipping {pinhole_id} (cam{cam_key}): fisheye camera")
            continue

        # Parse intrinsic matrix (flattened 3x3 row-major)
        intri = cam_calib['intri']
        K = np.array(intri, dtype=np.float64).reshape(3, 3)

        # Parse distortion coefficients [k1, k2, p1, p2, k3]
        dist = np.array(cam_calib['distor'], dtype=np.float64)

        # Parse extrinsics: virtualLidar -> camera
        # P_cam = R @ P_world + t
        rvec = cam_calib['virtualLidarToCam']['rotate']
        tvec = cam_calib['virtualLidarToCam']['trans']
        R = rodrigues_to_rotmat(rvec)
        t = np.array(tvec, dtype=np.float64)

        # Find all images for this camera
        img_dir = os.path.join(input_dir, 'img', pinhole_id)
        if not os.path.isdir(img_dir):
            print(f"  Warning: image directory {img_dir} not found, skipping")
            continue

        img_files = sorted([f for f in os.listdir(img_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        if not img_files:
            print(f"  Warning: no images found in {img_dir}, skipping")
            continue

        colmap_cam_id += 1

        for img_file in img_files:
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Warning: failed to read {img_path}")
                continue

            h, w = img.shape[:2]

            # Undistort image
            img_undist, K_new = undistort_image_and_intrinsics(img, K, dist)

            # Output image name: pinhole0_<timestamp>.png
            stem = os.path.splitext(img_file)[0]
            timestamp = stem.split('_')[-1]
            out_name = f"{pinhole_id}_{timestamp}.png"

            # Save undistorted image
            out_path = os.path.join(output_dir, 'images', out_name)
            cv2.imwrite(out_path, img_undist)

            # Store camera data
            key = f"{pinhole_id}_{timestamp}"
            cameras[key] = {
                'K_orig': K,
                'K_new': K_new,
                'dist': dist,
                'R': R,
                't': t,
                'width': w,
                'height': h,
                'pinhole_id': pinhole_id,
                'cam_key': cam_key,
                'image_name': out_name,
                'colmap_cam_id': colmap_cam_id,
            }
            images[key] = cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB)

            print(f"  {pinhole_id} (cam{cam_key}): {img_file} -> {out_name}"
                  f"  fx={K_new[0,0]:.1f} fy={K_new[1,1]:.1f}")

    print(f"\nTotal: {len(cameras)} images from {colmap_cam_id} cameras")

    # ============================================================
    # Step 2: Write COLMAP text files
    # ============================================================
    print("\n=== Step 2: Writing COLMAP files ===")

    cameras_txt = os.path.join(output_dir, 'sparse', '0', 'cameras.txt')
    write_cameras_txt(cameras, cameras_txt)
    print(f"  Written {cameras_txt}")

    images_txt = os.path.join(output_dir, 'sparse', '0', 'images.txt')
    write_images_txt(cameras, images_txt)
    print(f"  Written {images_txt}")

    # Empty points3D.txt (we use PLY directly)
    points3d_txt = os.path.join(output_dir, 'sparse', '0', 'points3D.txt')
    with open(points3d_txt, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0 (using PLY file instead)\n")
    print(f"  Written {points3d_txt}")

    # ============================================================
    # Step 3: Load and color LiDAR point cloud
    # ============================================================
    print("\n=== Step 3: Processing LiDAR point cloud ===")

    pcd_files = sorted(glob.glob(os.path.join(input_dir, '*.pcd')))
    if not pcd_files:
        print("ERROR: No .pcd files found in input directory!")
        sys.exit(1)

    # Read all PCD files and merge
    all_points = []
    all_intensities = []
    for pcd_file in pcd_files:
        pts, ints = read_pcd_ascii(pcd_file)
        all_points.append(pts)
        all_intensities.append(ints)
        print(f"  Loaded {len(pts)} points from {os.path.basename(pcd_file)}")

    points = np.vstack(all_points)
    intensities = np.concatenate(all_intensities)
    print(f"  Total: {len(points)} LiDAR points")

    # Color point cloud by projecting onto camera images
    print("  Coloring point cloud...")
    colors, colored_mask = color_pointcloud(points, cameras, images)
    print(f"  {colored_mask.sum()} / {len(points)} points visible in at least one camera")

    # Filter to visible points if requested
    if args.filter_visible:
        out_points = points[colored_mask]
        out_colors = colors[colored_mask]
        print(f"  Keeping {len(out_points)} visible points")
    else:
        out_points = points
        out_colors = colors
        print(f"  Keeping all {len(out_points)} points")

    # Write PLY
    ply_path = os.path.join(output_dir, 'sparse', '0', 'points3D.ply')
    write_ply(ply_path, out_points, out_colors)
    print(f"  Written {ply_path}")

    # ============================================================
    # Step 4: Generate depth maps using Depth Anything V2
    # ============================================================
    print("\n=== Step 4: Generating depth maps ===")

    if args.skip_depth:
        print("  Skipped (--skip_depth). Generate depth maps later with:")
        print(f"    python scripts/generate_depth_car_road.py --images_dir {os.path.join(output_dir, 'images')} --output_dir {os.path.join(output_dir, 'depth_maps')}")
    else:
        print("  LiDAR projection is too sparse (~12% pixel coverage) for DNGaussian.")
        print("  Generating dense monocular depth maps using Depth Anything V2...")
        depth_maps_dir = os.path.join(output_dir, 'depth_maps')
        images_dir = os.path.join(output_dir, 'images')
        generate_depth_maps(images_dir, depth_maps_dir)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print(f"Output: {output_dir}")
    print(f"  images/       : {len(cameras)} undistorted images")
    print(f"  sparse/0/     : COLMAP text files + colored PLY ({len(out_points)} points)")
    print(f"  depth_maps/   : Dense monocular depth maps")
    print("=" * 60)
    print(f"\nTo train DNGaussian (LLFF mode, all {len(cameras)} views):")
    print(f"  python train_llff.py \\")
    print(f"      -s {output_dir} \\")
    print(f"      --model_path output/car_road/scene1 \\")
    print(f"      -r 1 --eval --n_sparse 3")
    print(f"\nNote: With 4 images and llffhold=8, image 0 will be test, 1-3 will be train.")


if __name__ == '__main__':
    main()
