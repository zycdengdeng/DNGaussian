# 车路协同数据适配说明

## 概述

本项目将路侧多相机采集的场景数据转换为 DNGaussian 可训练的格式，训练得到 3D Gaussian Splatting 模型后，可从任意车端相机视角渲染出对应的 RGB 图像和深度图。

整体流程分为三个阶段：

```
原始路侧数据 ──(1)──> COLMAP 格式 ──(2)──> 3DGS 模型 ──(3)──> 车端视角渲染
```

---

## 坐标系定义

| 坐标系 | 说明 |
|--------|------|
| **World (virtualLiDAR)** | 路侧标定统一坐标系，即 calib.json 中的 virtualLidar 坐标系，也是 COLMAP 世界坐标系 |
| **LiDAR (车端)** | 车端 LiDAR 坐标系，通过 `world2lidar` 变换与 World 关联 |
| **Camera (车端)** | 车端各相机坐标系，通过外参 `cam2lidar` 与车端 LiDAR 关联 |

---

## 阶段一：路侧数据准备

**脚本：** `scripts/prepare_car_road.py`

### 输入

```
self_Dataset_sceneX/
├── calib.json                # 路侧标定文件（内参、畸变、外参）
├── img/
│   ├── pinhole0/             # cam3 图像
│   ├── pinhole1/             # cam6 图像
│   ├── pinhole2/             # cam9 图像
│   └── pinhole3/             # cam0 图像
└── *.pcd                     # LiDAR 点云（virtualLiDAR 坐标系）
```

**`calib.json` 关键字段：**

```json
{
  "camera": {
    "3": {
      "intri": [fx, 0, cx, 0, fy, cy, 0, 0, 1],   // 3x3 内参矩阵（行展开）
      "distor": [k1, k2, p1, p2, k3],               // 畸变系数
      "isFish": 0,                                    // 0=针孔, 1=鱼眼
      "virtualLidarToCam": {
        "rotate": [rx, ry, rz],                       // Rodrigues 旋转向量
        "trans": [tx, ty, tz]                         // 平移向量
      }
    }
  }
}
```

相机映射关系：`pinhole0→cam3, pinhole1→cam6, pinhole2→cam9, pinhole3→cam0`

### 处理步骤

1. **图像去畸变**：使用 `cv2.undistort()` 去畸变，同时计算新内参 `K_new`
2. **写入 COLMAP 文件**：
   - `cameras.txt` — PINHOLE 模型（fx, fy, cx, cy）
   - `images.txt` — 旋转四元数 + 平移（world→camera 变换）
   - `points3D.txt` — 空文件
3. **点云着色**：将 LiDAR 点投影到各相机图像上取色，保留可见点
4. **导出 PLY**：着色后的点云写入 `points3D.ply`

### 输出

```
data/car_road/sceneX/
├── images/                   # 去畸变后的图像
│   ├── pinhole0_<ts>.png
│   ├── pinhole1_<ts>.png
│   ├── pinhole2_<ts>.png
│   └── pinhole3_<ts>.png
├── sparse/0/
│   ├── cameras.txt           # COLMAP 相机内参
│   ├── images.txt            # COLMAP 相机位姿
│   ├── points3D.txt          # 空（用 PLY 替代）
│   └── points3D.ply          # 着色 LiDAR 点云
└── depth_maps/               # 深度图（阶段 1.5）
```

### 用法

```bash
python scripts/prepare_car_road.py \
    --input_dir self_Dataset_scene1 \
    --output_dir data/car_road/scene1 \
    --skip_depth
```

---

## 阶段 1.5：深度图生成

DNGaussian 训练需要单目深度先验。支持两种来源：

### 方式一：Depth Anything V3 (推荐)

外部运行 DA3 得到 metric depth `.npy` 文件，然后转换：

```bash
python scripts/convert_da3_depth.py \
    --depth_dir data/car_road/scene1/depth \
    --images_dir data/car_road/scene1/images \
    --output_dir data/car_road/scene1/depth_maps
```

**转换逻辑：**
- DA3 输出：`float32` 度量深度（远=大值）
- DNGaussian 输入：`uint8` 反深度 PNG（近=亮，远=暗）
- 训练时代码会做 `255.0 - depth_mono` 翻转

### 方式二：MiDaS DPT-Large

不加 `--skip_depth` 时 `prepare_car_road.py` 自动生成（相对深度，非度量深度）。

### 深度图命名规范

```
depth_maps/
├── depth_pinhole0_<ts>.png
├── depth_pinhole1_<ts>.png
├── depth_pinhole2_<ts>.png
└── depth_pinhole3_<ts>.png
```

命名规则：`depth_` + 对应图像文件名。

---

## 阶段二：训练 3DGS 模型

```bash
python train_llff.py \
    -s data/car_road/scene1 \
    --model_path output/car_road/scene1 \
    -r 1 --eval --n_sparse 3
```

使用 LLFF 模式（前向场景），4 张图中 image 0 做测试，image 1-3 做训练。

---

## 阶段三：车端视角渲染

**脚本：** `render_vehicle.py`

### 坐标变换链

```
Vehicle Camera ──(cam2lidar 外参)──> 车端 LiDAR ──(inv world2lidar)──> World (COLMAP)
```

用公式表示：

```
T_cam2world = T_lidar2world @ T_cam2lidar
            = inv(T_world2lidar) @ T_cam2lidar

T_world2cam = inv(T_cam2world)
```

### 输入文件

**车端标定文件（YAML）：**

```yaml
# camera_01_intrinsics.yaml
K: [fx, 0, cx, 0, fy, cy, 0, 0, 1]    # 3x3 内参
D: [k1, k2, p1, p2, k3, ...]            # 畸变系数

# camera_01_extrinsics.yaml
transform:
  rotation: {x: qx, y: qy, z: qz, w: qw}   # 四元数 (cam→lidar)
  translation: {x: tx, y: ty, z: tz}          # 平移 (cam→lidar)
```

**world2lidar 变换 JSON：**

```json
[
  {
    "timestamp": 1743583131842,
    "world2lidar": {
      "rotation": [rx, ry, rz],         // Rodrigues 旋转向量
      "translation": [tx, ty, tz]
    }
  }
]
```

### 车端相机列表

| ID | 名称 | 描述 | 分辨率 |
|----|------|------|--------|
| 1 | FN | 前视窄角 30° | 3840x2160 |
| 2 | FW | 前视广角 120° | 3840x2160 |
| 3 | FL | 左前视 120° | 3840x2160 |
| 4 | FR | 右前视 120° | 3840x2160 |
| 5 | RL | 左后视 60° | 1920x1080 |
| 6 | RR | 右后视 60° | 1920x1080 |
| 7 | RN | 后视 60° | 1920x1080 |

> 鱼眼相机（ID 2, 3, 4）使用 `cv2.fisheye` 去畸变，其余使用标准模型。

### 处理流程

1. 加载训练好的 3DGS 模型（`chkpnt_latest.pth`）
2. 根据时间戳查找最近的 `world2lidar` 变换
3. 对每个车端相机：
   - 加载内参 K、畸变 D、外参 R_cam2lidar / t_cam2lidar
   - 计算去畸变后的新内参 `new_K`
   - 计算相机在 COLMAP 世界坐标系下的位姿（`R_stored`, `T_stored`）
   - 构造虚拟 Camera 对象
   - 用 3DGS 渲染 RGB、深度、alpha

### 用法

```bash
python render_vehicle.py \
    --model_path output/car_road/scene1 \
    --vehicle_calib /path/to/vehicle/calibration/ \
    --transform_json /path/to/world2lidar.json \
    --timestamp 1743583131842 \
    --camera_ids 1 2 3 4 5 6 7 \
    --render_scale 4
```

### 输出

```
output/car_road/scene1/vehicle_renders/
├── FN/
│   ├── render.png          # RGB 渲染
│   ├── depth.png           # 深度可视化
│   ├── alpha.png           # 不透明度
│   └── camera_info.json    # 相机元数据
├── FW/
├── FL/
├── FR/
├── RL/
├── RR/
└── RN/
```

---

## 附：LiDAR 点云直接投影到车端（不经过 3DGS）

**脚本：** `road_car_transfer/undistort_projection_multithread_v2.py`

这是一个独立的工具，直接将世界坐标系下的 LiDAR 点云投影到车端相机生成深度图，不需要训练 3DGS 模型。

**变换流程：**

```
World 点云 ──(world2lidar)──> LiDAR 坐标系
           ──(inv cam2lidar)──> Camera 坐标系
           ──(new_K 投影)──> 图像坐标系
```

**批量处理：**

```bash
python road_car_transfer/run_batch_v2.py \
    --roadside-calib <路侧标定> \
    --vehicle-calib <车端标定> \
    --gt-images <车端真值图像> \
    --pcd <点云文件> \
    --transform-json <world2lidar JSON> \
    --output-dir <输出目录> \
    --timestamp <时间戳ms>
```

输出包括：深度图（`.npy` + `.jpg`）、去畸变真值图、对比图、叠加图。

---

## 完整工作流示例

```bash
# 1. 路侧数据转 COLMAP 格式
python scripts/prepare_car_road.py \
    --input_dir self_Dataset_scene1 \
    --output_dir data/car_road/scene1 \
    --skip_depth

# 2. 生成深度图（DA3 度量深度转换）
python scripts/convert_da3_depth.py \
    --depth_dir data/car_road/scene1/depth \
    --images_dir data/car_road/scene1/images \
    --output_dir data/car_road/scene1/depth_maps

# 3. 训练 3DGS
python train_llff.py \
    -s data/car_road/scene1 \
    --model_path output/car_road/scene1 \
    -r 1 --eval --n_sparse 3

# 4. 从车端视角渲染
python render_vehicle.py \
    --model_path output/car_road/scene1 \
    --vehicle_calib /path/to/vehicle/calibration/ \
    --transform_json /path/to/world2lidar.json \
    --timestamp 1743583131842 \
    --camera_ids 1 2 3 4 5 6 7
```
