# Car-Road 路侧-车端场景重建

使用路侧相机（roadside cameras）训练 3DGS 场景，然后从车端相机（vehicle cameras）视角渲染。

## 数据准备

原始数据结构（`self_Dataset/`）：

```
self_Dataset/
├── calib.json                    # 路侧相机标定（virtualLidar 坐标系）
├── 1743583131842.pcd             # LiDAR 点云
└── img/
    ├── pinhole0/                 # 路侧相机0 (对应 calib.json 中 cam3)
    │   └── 1743583131842.png
    ├── pinhole1/                 # 路侧相机1 (对应 cam6)
    │   └── 1743583131842.png
    ├── pinhole2/                 # 路侧相机2 (对应 cam9)
    │   └── 1743583131842.png
    └── pinhole3/                 # 路侧相机3 (对应 cam0)
        └── 1743583131842.png
```

### Step 1: 转换为 COLMAP 格式

```bash
python scripts/prepare_car_road.py \
    --input_dir self_Dataset \
    --output_dir data/car_road/scene1
```

输出结构：

```
data/car_road/scene1/
├── images/                       # 去畸变后的图像
│   ├── pinhole0_1743583131842.png
│   ├── pinhole1_1743583131842.png
│   ├── pinhole2_1743583131842.png
│   └── pinhole3_1743583131842.png
├── depth_maps/                   # LiDAR 投影深度图
│   ├── depth_pinhole0_1743583131842.png
│   └── ...
├── sparse/0/
│   ├── cameras.txt               # COLMAP 相机内参
│   ├── images.txt                # COLMAP 相机外参
│   ├── points3D.ply              # 带颜色的 LiDAR 点云
│   └── points3D.txt              # 空文件（占位）
└── depth/                        # MiDaS 单目深度（Step 2 生成）
```

### Step 2: 生成单目深度图

DNGaussian 需要单目深度先验作为正则化。使用 MiDaS 生成：

```bash
# 方法一：本地 MiDaS（推荐，离线可用）
git clone https://github.com/isl-org/MiDaS.git third_party/MiDaS
# 下载 dpt_large_384.pt 到 third_party/MiDaS/weights/
python scripts/generate_depth_car_road.py \
    --images_dir data/car_road/scene1/images \
    --output_dir data/car_road/scene1/depth \
    --midas_path third_party/MiDaS

# 方法二：HuggingFace 镜像（国内服务器）
python scripts/generate_depth_car_road.py \
    --images_dir data/car_road/scene1/images \
    --output_dir data/car_road/scene1/depth \
    --hf_mirror

# 方法三：torch.hub（需要访问 GitHub）
python scripts/generate_depth_car_road.py \
    --images_dir data/car_road/scene1/images \
    --output_dir data/car_road/scene1/depth
```

每张图像生成两个文件：`{name}_depth.npy`（原始深度）和 `{name}_depth.png`（可视化）。

## 训练

```bash
bash scripts/run_car_road.sh data/car_road/scene1 output/car_road/scene1 0
```

参数说明：
- 参数1：数据路径
- 参数2：模型输出路径
- 参数3：GPU 编号

训练使用 4 张路侧相机图像，6000 次迭代。训练完成后自动运行渲染和指标评估。

**输出**：

```
output/car_road/scene1/
├── chkpnt_latest.pth             # 模型权重
├── point_cloud/
│   └── iteration_6000/
│       └── point_cloud.ply       # 训练后的高斯点云
├── train/ours_6000/              # 训练视角渲染结果
│   ├── renders/                  # 渲染图像
│   └── gt/                       # 对应的 GT 图像
└── results.json                  # PSNR/SSIM/LPIPS 指标
```

## 车端视角渲染

训练完成后，使用 `render_vehicle.py` 从车端相机视角渲染 3DGS 场景：

```bash
python render_vehicle.py \
    --model_path output/car_road/scene1 \
    --vehicle_calib /path/to/vehicle/calibration/ \
    --transform_json /path/to/world2lidar.json \
    --timestamp 1743583131842 \
    --camera_ids 1 2 3 4 5 6 7 \
    --render_scale 4
```

或者通过 `run_car_road.sh` 一键完成训练+渲染：

```bash
bash scripts/run_car_road.sh \
    data/car_road/scene1 \
    output/car_road/scene1 \
    0 \
    /path/to/vehicle/calibration \
    /path/to/world2lidar.json \
    1743583131842
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--model_path` | 训练好的模型路径 |
| `--vehicle_calib` | 车端标定文件夹，包含 `camera_XX_intrinsics.yaml` 和 `camera_XX_extrinsics.yaml` |
| `--transform_json` | 车端 LiDAR → 路侧 virtualLidar 的 world2lidar 变换（Rodrigues 旋转 + 平移） |
| `--timestamp` | 时间戳（毫秒），用于匹配 world2lidar 变换 |
| `--camera_ids` | 要渲染的车端相机编号，默认 1-7（全部） |
| `--render_scale` | 分辨率缩放因子，默认 4（3840x2160 → 960x540） |

### 车端相机编号

| ID | 名称 | 视角 | 分辨率 |
|----|------|------|--------|
| 1 | FN | 前窄 30° | 3840x2160 |
| 2 | FW | 前宽 120° | 3840x2160 |
| 3 | FL | 左前 120° | 3840x2160 |
| 4 | FR | 右前 120° | 3840x2160 |
| 5 | RL | 左后 60° | 1920x1080 |
| 6 | RR | 右后 60° | 1920x1080 |
| 7 | RN | 后窄 60° | 1920x1080 |

### 车端标定文件格式

**intrinsics** (`camera_XX_intrinsics.yaml`)：
```yaml
K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]    # 3x3 内参矩阵（行优先）
D: [k1, k2, p1, p2, k3]                 # 畸变系数
```

**extrinsics** (`camera_XX_extrinsics.yaml`)：
```yaml
transform:
  rotation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0          # 四元数 (x,y,z,w)，camera → lidar
  translation:
    x: 0.0
    y: 0.0
    z: 0.0          # 平移，camera → lidar
```

**world2lidar** (`world2lidar.json`)：
```json
[
  {
    "timestamp": 1743583131842,
    "world2lidar": {
      "rotation": [rx, ry, rz],          // Rodrigues 旋转向量
      "translation": [tx, ty, tz]
    }
  }
]
```

### 输出结果

```
output/car_road/scene1/vehicle_renders/
├── FN/                 # 前窄相机
│   ├── render.png      # RGB 渲染图
│   ├── depth.png       # 深度图
│   ├── alpha.png       # 透明度图（可见区域）
│   └── camera_info.json
├── FW/
├── FL/
├── FR/
├── RL/
├── RR/
└── RN/
```

## 坐标系说明

整个流程涉及三个坐标系：

1. **virtualLidar 坐标系**（路侧）：3DGS 场景的世界坐标系，即 COLMAP 的世界系
2. **车端 LiDAR 坐标系**：车载 LiDAR 的坐标系
3. **车端相机坐标系**：各车载相机的坐标系

变换链：
```
车端相机 --(extrinsics: cam2lidar)--> 车端LiDAR --(inv world2lidar)--> virtualLidar (世界系)
```

**注意**：3DGS 训练不会改变坐标系。训练后高斯点的 `_xyz` 始终在 virtualLidar 坐标系中。

## 已知限制

- 路侧相机只有 4 个视角（高处俯视），视角稀疏
- 从与训练视角差异很大的车端视角渲染时，效果可能受限于 few-shot 重建的深度精度
- 训练过程中高斯点可能发生深度漂移（从训练视角看正确，但 3D 位置偏移）
