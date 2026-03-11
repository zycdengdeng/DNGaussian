# Car-Road 路侧数据集 - 3DGS 适配技术规格

本文档完整描述了路侧相机数据集的结构、坐标系、标定、点云、深度图等所有信息，
以及如何将其适配为 3DGS（Gaussian Splatting）训练所需的 COLMAP 格式。
可直接用于适配 Street Gaussians (S3GS) 等其他 3DGS 框架。

---

## 1. 原始数据结构

```
self_Dataset/
├── calib.json                    # 标定文件（所有相机 + 所有雷达）
├── 1743583131842.pcd             # LiDAR 点云（ASCII PCD）
└── img/
    ├── pinhole0/                 # 路侧 pinhole 相机 0
    │   └── 1743583131842.png     # 1280x720 RGB
    ├── pinhole1/
    │   └── 1743583131842.png
    ├── pinhole2/
    │   └── 1743583131842.png
    └── pinhole3/
        └── 1743583131842.png
```

**关键事实**：
- 只有 **1 帧**（时间戳 `1743583131842`），4 个 pinhole 相机
- 所有 4 张图像分辨率 **1280 x 720**
- PCD 文件是 ASCII 格式，314705 个点，字段: `x y z intensity`
- calib.json 中还有鱼眼相机（isFish=1），但我们**只用 pinhole 相机**（isFish=0）

---

## 2. 坐标系定义

整个系统有一个统一的世界坐标系：**virtualLidar 坐标系**。

- 这是一个由标定系统定义的虚拟坐标系
- 所有相机外参和点云都在此坐标系下
- PCD 点云的坐标已经在 virtualLidar 系中
- 3DGS 训练后，高斯点的 `_xyz` 坐标直接就是 virtualLidar 系（**训练过程不做任何坐标变换**）

### 坐标范围

| | X | Y | Z |
|---|---|---|---|
| 点云（原始 PCD，314705 点） | [-507.7, 256.9] | [-478.4, 368.0] | [-9.2, 46.0] |
| 点云（可见性过滤后，287667 点） | [-507.7, 256.9] | [-478.4, 368.0] | [-9.2, 9.9] |
| 点云中心 | -62.0 | -13.9 | -1.4 |
| 相机中心（4个） | [0.3, -129.3] | [-68.4, 42.8] | [3.8, 4.5] |
| 相机中心均值 | -63.5 | -12.1 | 4.2 |

**空间含义**：
- Z ≈ 0 是地面
- Z ≈ 4 是相机高度（路侧杆约 4m）
- XY 平面覆盖约 760m x 846m 的区域（但相机仅覆盖中心约 130m x 110m）
- 场景尺度很大（单位是米）

---

## 3. calib.json 详细结构

```json
{
  "data": "data-1",
  "imgSize": {
    "fish": [1280, 1280],
    "notFish": [1280, 720]
  },
  "lidar": {
    "0": {
      "name": "rad0_1728605893473.pcd",
      "lidarToVirtualLidar": {
        "rotateMatrix": [9个浮点数, 行优先 3x3 旋转矩阵],
        "trans": [tx, ty, tz]
      }
    }
    // ... lidar 0-3
  },
  "camera": {
    "0": {
      "isFish": 0,           // 0=pinhole, 1=fisheye
      "intri": [9个浮点数],   // 行优先 3x3 内参矩阵 [fx,0,cx, 0,fy,cy, 0,0,1]
      "distor": [k1,k2,p1,p2,k3],  // OpenCV 畸变系数（pinhole 5个）
      "virtualLidarToCam": {
        "rotate": [rx, ry, rz],     // Rodrigues 旋转向量（3维）
        "trans": [tx, ty, tz]       // 平移向量
      }
    }
    // ... camera 0,2,3,5,6,8,9,11
  }
}
```

### 相机编号映射

图像文件夹名 → calib.json 中的 camera key:

| 文件夹 | camera key | isFish | 说明 |
|--------|-----------|--------|------|
| pinhole0 | cam3 | 0 | pinhole |
| pinhole1 | cam6 | 0 | pinhole |
| pinhole2 | cam9 | 0 | pinhole |
| pinhole3 | cam0 | 0 | pinhole |

**注意**：这个映射关系是硬编码的，不是自动推导的。

### 外参含义

`virtualLidarToCam` 是 **world-to-camera** 变换：

```
P_camera = R @ P_virtualLidar + t
```

其中 R 由 Rodrigues 旋转向量通过 `cv2.Rodrigues()` 转换为 3x3 矩阵。

### 4 个 pinhole 相机的具体参数

**内参**（去畸变后的 K_new，由 `cv2.getOptimalNewCameraMatrix(K, dist, (w,h), alpha=0)` 计算）：

| 相机 | fx | fy | cx | cy |
|------|----|----|----|----|
| pinhole0 (cam3) | 1158.10 | 1243.57 | 608.39 | 365.47 |
| pinhole1 (cam6) | 1108.43 | 1249.98 | 632.13 | 385.59 |
| pinhole2 (cam9) | 1131.80 | 1260.26 | 670.15 | 362.35 |
| pinhole3 (cam0) | 1102.69 | 1228.19 | 635.85 | 376.51 |

**注意**：fx ≠ fy（非正方形像素），这在 COLMAP PINHOLE 模型中是正常的。

**外参**（virtualLidarToCam，即 world-to-camera）：

| 相机 | Rodrigues 旋转向量 | 平移 t |
|------|--------------------|--------|
| pinhole0 (cam3) | [1.419, 1.389, -1.033] | [9.473, 4.205, 1.819] |
| pinhole1 (cam6) | [2.927, -0.012, 0.115] | [49.583, -18.720, 74.599] |
| pinhole2 (cam9) | [-1.388, -1.430, 1.079] | [-18.859, -30.364, 125.038] |
| pinhole3 (cam0) | [0.060, 0.046, -3.079] | [-69.270, -6.279, 34.073] |

**相机中心在 virtualLidar 系中的位置**（`center = -R^T @ t`）：

| 相机 | X | Y | Z | 说明 |
|------|---|---|---|------|
| pinhole0 | 0.294 | -9.546 | 4.417 | 最北偏东 |
| pinhole1 | -60.629 | -68.435 | 3.830 | 东南方向 |
| pinhole2 | -129.293 | -13.244 | 4.483 | 最西 |
| pinhole3 | -64.441 | 42.790 | 3.881 | 北侧 |

4 个相机分布在约 130m x 110m 的范围内，高度约 4m（路侧杆顶）。

---

## 4. 点云处理

### PCD 读取

```python
# PCD 是 ASCII 格式
# FIELDS x y z intensity
# SIZE 4 4 4 4
# TYPE F F F F
# POINTS 314705
# DATA ascii
# 每行: x y z intensity

points, intensities = read_pcd_ascii("1743583131842.pcd")
# points: (314705, 3) float64, 在 virtualLidar 坐标系中
# intensities: (314705,) float64
```

### 点云着色

将 3D 点投影到每个相机上获取 RGB 颜色：

```python
# 对每个相机:
# 1. P_cam = R @ P_world + t  (用去畸变后的 K_new)
# 2. pixel_u = fx * Xc/Zc + cx
# 3. pixel_v = fy * Yc/Zc + cy
# 4. 如果多个相机看到同一个点，取深度最小（最近）的相机颜色
# 5. 只保留 depth > 0.5m 且在图像范围内的投影

colors, colored_mask = color_pointcloud(points, cameras, images)
# 287667 / 314705 (91.4%) 点被至少一个相机着色
```

### 可见性过滤

**默认开启**（`--filter_visible`）：只保留被至少一个相机看到的点。

过滤后点云：287667 点，Z 范围从 [-9.2, 46.0] 缩小到 [-9.2, 9.9]。

---

## 5. 适配为 COLMAP 格式

### 输出目录结构

```
data/car_road/scene1/
├── images/                                  # 去畸变后的 RGB 图像
│   ├── pinhole0_1743583131842.png          # 1280x720 RGB
│   ├── pinhole1_1743583131842.png
│   ├── pinhole2_1743583131842.png
│   └── pinhole3_1743583131842.png
├── depth_maps/                              # 深度先验（8-bit 灰度 PNG）
│   ├── depth_pinhole0_1743583131842.png    # 由 LiDAR 投影生成
│   ├── depth_pinhole1_1743583131842.png
│   ├── depth_pinhole2_1743583131842.png
│   └── depth_pinhole3_1743583131842.png
└── sparse/0/
    ├── cameras.txt                          # COLMAP 相机内参
    ├── images.txt                           # COLMAP 相机外参
    ├── points3D.ply                         # 带颜色的 LiDAR 点云（287667 点）
    └── points3D.txt                         # 空文件（占位）
```

### cameras.txt 格式

```
# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# PINHOLE 模型参数: fx, fy, cx, cy
1 PINHOLE 1280 720 1158.097389 1243.568565 608.388069 365.474083
2 PINHOLE 1280 720 1108.432852 1249.984942 632.125686 385.589687
3 PINHOLE 1280 720 1131.799679 1260.256753 670.146921 362.352168
4 PINHOLE 1280 720 1102.686843 1228.191536 635.852773 376.510887
```

每个 pinhole 相机一个 CAMERA_ID。**参数是去畸变后的 K_new**，不是原始 K。

### images.txt 格式

```
# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
# 后面跟空行（无 2D 特征点）
1 0.4364477893 0.5704426931 0.5581423033 -0.4154342673 9.4729739600 4.2051113000 1.8188053800 1 pinhole0_1743583131842.png

2 0.5971664233 0.7984724965 -0.0647725855 0.0404777392 49.5832708000 -18.7197268000 74.5989277900 2 pinhole1_1743583131842.png

3 0.4203732198 0.5489449457 -0.5781290203 0.4332581661 -18.8586402300 -30.3637046400 125.0379989900 3 pinhole2_1743583131842.png

4 0.0396522435 0.0478754012 0.7981974685 -0.5991798117 -69.2700000000 -6.2791345100 34.0731939400 4 pinhole3_1743583131842.png

```

- **四元数 (QW, QX, QY, QZ)** + **平移 (TX, TY, TZ)** = world-to-camera 变换
- 四元数由 Rodrigues → 旋转矩阵 → COLMAP 四元数格式 (`rotmat2qvec`) 转换
- 平移就是 calib.json 中的 `virtualLidarToCam.trans`
- COLMAP 约定：`R_w2c = qvec2rotmat(qvec)`, `t_w2c = tvec`, 相机中心 = `-R_w2c^T @ t_w2c`

### points3D.ply 格式

```
ply
format binary_little_endian 1.0
element vertex 287667
property float x        # virtualLidar 坐标
property float y
property float z
property float nx       # 法线（全 0）
property float ny
property float nz
property uchar red      # 从图像采样的颜色 [0, 255]
property uchar green
property uchar blue
end_header
```

**重要**：这个 PLY 就是 3DGS 初始化用的点云。DNGaussian 的 `fetchPly()` 读取时会把颜色除以 255 归一化到 [0, 1]。

---

## 6. 深度图

### 来源和格式

**目前实际使用的**是 `depth_maps/` 目录下的 LiDAR 投影深度图。

生成方式：
1. 将 LiDAR 3D 点投影到每个相机
2. 生成稀疏深度图（同一像素取最近的点）
3. 用形态学膨胀迭代填充（15 次迭代，5x5 椭圆核）
4. 转为反深度（disparity）格式的 8-bit PNG

**深度 PNG 约定**：
- **像素值 255 = 近**（高 disparity）
- **像素值 0 = 远**（低 disparity）
- 无效区域（无 LiDAR 数据）= 0

**训练时的使用**（`train_llff.py` 第 104 行）：
```python
depth_mono = 255.0 - viewpoint_cam.depth_mono
# 反转后: 0 = 近, 255 = 远（与渲染深度方向一致）
# 然后用 patch_norm_mse_loss 与渲染深度做归一化 MSE
```

### 文件命名

训练代码（`dataset_readers.py` 第 126 行）读取路径为：
```
{source_path}/depth_maps/depth_{image_stem}.png
```

例如图像 `images/pinhole0_1743583131842.png` 对应：
```
depth_maps/depth_pinhole0_1743583131842.png
```

### 分辨率

当前 depth_maps 中的深度图分辨率是 **504 x 280**（不是 1280x720），
但训练时 `PILtoTorch` 会自动 resize 到训练分辨率。

---

## 7. DNGaussian 读取流程

### 入口

`scene/__init__.py` → `readColmapSceneInfo()` → `readColmapCameras()`

1. 检测到 `sparse/` 目录 → 使用 COLMAP 加载器
2. 先尝试 `.bin`，失败则读 `.txt`（我们用 txt）
3. `readColmapCameras()`:
   - 从 images.txt 读 R（qvec → rotmat → **转置存储**）和 T
   - 从 cameras.txt 读 PINHOLE 内参 → 计算 FoVX/FoVY
   - 加载图像和 depth_maps
4. 如果无 `--eval`：所有 4 张图片作为 train，无 test/eval
5. 点云加载：直接读 `sparse/0/points3D.ply`

### R 的存储约定

**关键**：DNGaussian/3DGS 中 R 存储为 **转置** 形式：

```python
# dataset_readers.py line 105
R = np.transpose(qvec2rotmat(extr.qvec))
# 存储的 R = R_w2c^T

# cameras.py 中还原:
# world_view_transform = getWorld2View2(R, T)
# getWorld2View2 内部会转置回来得到 R_w2c
```

所以 CameraInfo 中：
- `R` = `R_w2c^T`（3x3，转置存储）
- `T` = `t_w2c`（3维平移）
- 相机中心 = `-R^T @ T`（注意这里 R 已经是转置的，所以 `R^T = R_w2c`）

### getNerfppNorm

计算场景归一化参数（但**只有 radius 被使用**，translate 被丢弃）：

```python
cameras_extent = radius = max_camera_distance_from_center * 1.1
# 对我们的场景: cameras_extent ≈ 72.4
# 仅用于 densification 的梯度阈值缩放
```

---

## 8. 训练配置

### 命令

```bash
python train_llff.py -s data/car_road/scene1 --model_path output/car_road/scene1 \
    -r 1 --iterations 6000 --lambda_dssim 0.2 \
    --densify_grad_threshold 0.0013 --prune_threshold 0.01 \
    --densify_until_iter 6000 --percent_dense 0.01 \
    --position_lr_init 0.016 --position_lr_final 0.00016 \
    --position_lr_max_steps 5500 --position_lr_start 500 \
    --split_opacity_thresh 0.1 --error_tolerance 0.00025 \
    --scaling_lr 0.003 \
    --shape_pena 0.002 --opa_pena 0.001 \
    --near 0
```

关键参数：
- **不使用 `--eval`**：4 张图全部用于训练（不分 train/test）
- `-r 1`：不缩放分辨率
- `--near 0`：近裁剪面为 0
- 未指定 `--rand_pcd` 和 `--mvs_pcd`：使用 LiDAR PLY 作为初始点云

### 输出

```
output/car_road/scene1/
├── cfg_args                      # 序列化的训练参数
├── chkpnt_latest.pth             # 最新 checkpoint
├── input.ply                     # 初始点云的副本
├── cameras.json                  # 相机参数 JSON
├── point_cloud/
│   └── iteration_6000/
│       └── point_cloud.ply       # 训练后的高斯点云
└── train/ours_6000/              # 渲染结果
    ├── renders/
    └── gt/
```

---

## 9. 已知问题与注意事项

### 深度漂移

只有 4 个高处俯视视角（Z≈4m），训练后高斯点存在严重深度漂移：
- 初始点云 Z 范围：[-9.2, 9.9]
- 训练后高斯 Z 范围：[-246, 156]（中位数从 ~0 漂移到 -48）

从训练视角渲染正常，但从其他视角（如车端地面视角）渲染效果差。
这是 few-shot 3DGS 的固有限制。

### 场景尺度

场景很大（相机间距 ~130m，点云范围 ~760m），单位是米。
某些 3DGS 实现可能需要缩放场景到合理范围。

### 无 poses_bounds.npy

DNGaussian 的 spiral 渲染路径需要 `poses_bounds.npy`（LLFF 格式），
我们没有生成。已修改 `train_llff.py` 使其在缺失时跳过 spiral 渲染。

### 图像命名规则

```
{pinhole_id}_{timestamp}.png
```
例如: `pinhole0_1743583131842.png`

depth 文件名在前面加 `depth_` 前缀:
```
depth_{pinhole_id}_{timestamp}.png
```

---

## 10. 适配其他 3DGS 框架的要点

要适配到 S3GS 或其他框架时，核心需要提供：

### 必需数据

1. **相机内参**（去畸变后）：4 个 PINHOLE 相机，fx/fy/cx/cy 见第 5 节
2. **相机外参**：world-to-camera 的 R（3x3 旋转矩阵）和 t（3 维平移），见 images.txt
3. **RGB 图像**：1280x720，已去畸变
4. **初始点云**：287667 个带颜色的 3D 点，在 virtualLidar 系中
5. **深度先验**（可选）：8-bit 灰度 PNG，disparity 格式（255=近，0=远）

### 外参转换

如果目标框架需要 camera-to-world（c2w）：
```python
# 从 COLMAP 的 w2c 转换
R_w2c = qvec2rotmat(qvec)    # 3x3
t_w2c = tvec                  # 3维
T_w2c = np.eye(4)
T_w2c[:3, :3] = R_w2c
T_w2c[:3, 3] = t_w2c
T_c2w = np.linalg.inv(T_w2c)
# camera_position = T_c2w[:3, 3]
# camera_rotation = T_c2w[:3, :3]
```

### 车端视角渲染

如果需要从车端相机视角渲染：

```python
# 变换链: 车端相机 → 车端LiDAR → virtualLidar(世界系)
T_c2w = inv(T_world2lidar) @ T_cam2lidar

# T_world2lidar: 来自 world2lidar.json (Rodrigues旋转 + 平移)
# T_cam2lidar: 来自车端标定 (四元数旋转 + 平移)
```

world2lidar.json 格式:
```json
[{
  "timestamp": 1743583131842,
  "world2lidar": {
    "rotation": [rx, ry, rz],       // Rodrigues 旋转向量
    "translation": [tx, ty, tz]
  }
}]
```

车端标定 YAML 格式:
```yaml
# camera_XX_extrinsics.yaml
transform:
  rotation: {x: 0, y: 0, z: 0, w: 1}   # 四元数 (x,y,z,w), cam→lidar
  translation: {x: 0, y: 0, z: 0}       # 平移, cam→lidar

# camera_XX_intrinsics.yaml
K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]      # 3x3 行优先
D: [k1, k2, p1, p2, k3]                  # 畸变系数
```
