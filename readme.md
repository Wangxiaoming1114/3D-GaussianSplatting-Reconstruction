# 物体 A 与背景场景的 3D Gaussian Splatting 重建

本仓库用于真实多视角三维重建部分，主要完成以下两个资产的构建：

1. **物体 A：黑色杯子的真实多视角 3DGS 重建**
2. **背景场景：Mip-NeRF 360 数据集中 treehill 场景的 3DGS 重建**

本部分使用 COLMAP 进行相机位姿估计与稀疏点云重建，并基于 3D Gaussian Splatting 完成三维表示训练和新视角渲染。最终得到的 `point_cloud.ply`、渲染图像和 COLMAP 重建结果将用于后续与 AIGC 生成物体 B/C 进行场景融合。

---

## 1. 仓库结构

当前 GitHub 仓库主要包含 3DGS 训练与渲染代码：

```text
GAUSSIAN-SPLATTING/
├── arguments/
├── assets/
├── gaussian_renderer/
├── lpipsPyTorch/
├── scene/
├── submodules/
├── utils/
├── convert.py
├── environment.yml
├── render.py
└── train.py
```

各文件和文件夹作用如下：

| 文件 / 文件夹             | 作用                          |
| -------------------- | --------------------------- |
| `convert.py`         | 使用 COLMAP 对多视角图像进行位姿估计与格式转换 |
| `train.py`           | 训练 3D Gaussian Splatting 模型 |
| `render.py`          | 使用训练好的 3DGS 模型进行新视角渲染       |
| `gaussian_renderer/` | 3DGS 渲染模块                   |
| `scene/`             | 数据读取、相机加载、场景初始化相关代码         |
| `arguments/`         | 训练与渲染参数配置                   |
| `utils/`             | 工具函数                        |
| `submodules/`        | 3DGS 依赖的 CUDA / C++ 扩展模块    |
| `environment.yml`    | Conda 环境配置文件                |

由于数据集、训练结果和模型权重文件较大，不直接上传到 GitHub，而是通过网盘链接或外部存储提供。

---

## 2. 实验任务说明

本仓库对应作业 Task 1 中的两个部分：

### 2.1 物体 A：真实多视角重建

物体 A 是一个真实拍摄的黑色杯子。实验流程如下：

```text
手机多视角拍摄
→ COLMAP 相机位姿估计
→ 稀疏点云初始化
→ 3DGS 训练
→ 新视角渲染
```

黑色杯子属于低纹理物体，表面颜色较深、局部特征较少，因此 COLMAP 匹配难度较高。实验中通过筛选清晰、连续、重叠度较高的图像，提高了图像注册效果。


---

### 2.2 背景场景：Mip-NeRF 360 treehill 重建

背景场景选用 Mip-NeRF 360 数据集中的 `treehill` 场景。该场景包含树木、地面、植被和远处环境，纹理较丰富，适合作为后续融合任务的统一背景。

实验流程如下：

```text
Mip-NeRF 360 treehill 多视角数据
→ COLMAP 相机位姿估计
→ 3DGS 训练
→ 背景新视角渲染
```


---

## 3. 环境配置

推荐使用 Linux + Conda 环境运行。

### 3.1 创建 Conda 环境

可以使用仓库中的 `environment.yml` 创建环境：

```bash
conda env create -f environment.yml
conda activate gaussian_splatting
```

如果环境创建较慢，也可以手动创建：

```bash
conda create -n gaussian_splatting python=3.8 -y
conda activate gaussian_splatting
```

然后安装常用依赖：

```bash
pip install torch torchvision torchaudio
pip install plyfile tqdm opencv-python pillow numpy
```

### 3.2 安装 3DGS 子模块

3DGS 需要安装 CUDA 扩展模块：

```bash
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

如果 `simple-knn` 子模块缺失，可以重新拉取：

```bash
git clone https://github.com/camenduru/simple-knn.git submodules/simple-knn
pip install ./submodules/simple-knn
```


请确保本机 CUDA 编译环境与 PyTorch CUDA 版本尽量匹配。

---

## 4. 数据准备

本实验包含两类数据：

```text
data/
├── object_A/
└── treehill/
```

---

### 4.1 物体 A 数据

物体 A 为黑色杯子。若已经完成 COLMAP 处理，推荐结构如下：

```text
data/object_A/
├── images/
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       ├── points3D.bin
│       ├── frames.bin
│       └── rigs.bin
└── stereo/
```

如果是未经过 COLMAP 的原始图片，推荐结构如下：

```text
data/object_A/
└── input/
    ├── 0001.jpg
    ├── 0002.jpg
    └── ...
```

然后使用 `convert.py` 进行 COLMAP 转换。

---

### 4.2 背景 treehill 数据

背景使用 Mip-NeRF 360 的 `treehill` 场景。推荐结构如下：

```text
treehill/
├── images/
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── stereo/
```

如果已有 COLMAP 结果，则可以直接训练 3DGS；如果只有原始图片，则需要先运行 `convert.py`。

---

## 5. COLMAP 数据转换

如果数据还没有 COLMAP 位姿，可以运行：

```bash
python convert.py -s data/object_A
```

或者对背景运行：

```bash
python convert.py -s treehill
```

本实验中使用的是 CPU 版本 COLMAP，因此对 `convert.py` 中的 COLMAP 参数进行了适配。主要设置包括：

```text
--FeatureExtraction.use_gpu 0
--FeatureMatching.use_gpu 0
--FeatureExtraction.num_threads 2
--FeatureExtraction.max_image_size 1600
--SiftExtraction.max_num_features 8192
```

这样可以降低 CPU 内存压力，避免大图像、多线程 SIFT 提取时被系统 kill。

完成后可以检查 COLMAP 重建质量：

```bash
colmap model_analyzer --path data/object_A/sparse/0
```

或者：

```bash
colmap model_analyzer --path treehill/sparse/0
```


---

## 6. 训练 3DGS

### 6.1 训练物体 A

```bash
python train.py \
  -s data/object_A \
  -m output/object_A_test \
  --iterations 30000 \
  --disable_viewer
```


注意：`-s` 指定的目录必须能被 3DGS 识别，通常需要包含：

```text
images/
sparse/0/
```


---

### 6.2 训练背景 treehill

```bash
python train.py \
  -s treehill \
  -m output/background_treehill_test \
  --iterations 30000 \
  --disable_viewer
```

---

## 7. 渲染结果

### 7.1 渲染物体 A

```bash
python render.py -m output/object_A_test --iteration 30000
```

渲染结果保存在：

```text
output/object_A_test/train/ours_30000/renders/
```

---

### 7.2 渲染背景 treehill

```bash
python render.py -m output/background_treehill_test --iteration 30000
```

渲染结果保存在：

```text
output/background_treehill_test/train/ours_30000/renders/
```

---

## 8. 输出资产说明

### 8.1 物体 A 输出

```text
output/object_A_test/
├── cfg_args
├── cameras.json
├── point_cloud/
│   └── iteration_30000/
│       └── point_cloud.ply
└── train/
    └── ours_30000/
        └── renders/
```

核心资产：

```text
output/object_A_test/point_cloud/iteration_30000/point_cloud.ply
```


---

### 8.2 背景输出

```text
output/background_treehill_test/
├── cfg_args
├── cameras.json
├── point_cloud/
│   └── iteration_30000/
│       └── point_cloud.ply
└── train/
    └── ours_30000/
        └── renders/
```

核心资产：

```text
output/background_treehill_test/point_cloud/iteration_30000/point_cloud.ply
```


---


## 9. 实验总结

本仓库完成了课程项目中真实多视角重建部分的实现。对于物体 A，本文使用手机拍摄的黑色杯子图像完成 COLMAP 位姿估计和 3DGS 重建；对于背景场景，本文使用 Mip-NeRF 360 数据集中的 `treehill` 场景完成背景 3DGS 重建。最终得到的 3DGS 资产将与 AIGC 生成的物体 B/C 一起用于最终场景融合与多视角漫游渲染。
