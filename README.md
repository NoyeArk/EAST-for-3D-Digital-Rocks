# EAST-for-3D-Digital-Rocks

本仓库是论文 “Efficiently Reconstructing High-Quality Details of 3D Digital Rocks with Super-Resolution Transformer”的官方 PyTorch 实现。

源码主要基于 [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch)。我们提供了完整的训练和测试代码。你可以从零开始训练模型，也可以使用预训练模型对数字岩心图像进行超分辨率重建。预训练模型近期将上线。

## 代码
### 依赖环境
* Python 3.8.5
* PyTorch = 2.0.1
* numpy
* cv2
* skimage
* tqdm

### 快速开始

```bash
git clone https://github.com/MHDXing/MASR-for-Digital-Rock-Images.git
cd EAST-for-3D-Digital-Rocks-main/src
```

## 数据集
我们使用的数据集来源于 [DeepRockSR-3D](https://www.digitalrocksportal.org/projects/215)。
其中共有 2400 张训练集、300 张测试集和 300 张验证集的高分辨率三维图像（100x100x100）。

#### 训练
1. 下载数据集并解压到任意位置，然后在 `./options.py` 或 `demo.sh` 中修改 `dir_data` 参数为你存放数据的位置。
2. 可以通过修改 `./options.py` 文件来自定义不同模型的超参数。
3. 运行 `main.py`，推荐使用脚本文件 `demo.sh`：
```bash
bash demo.sh
```
4. 如果在 `./options` 文件中的 `save` 参数设置为 `EAST`，则结果会保存在 `./experiments/EAST` 文件夹下。

#### 测试
1. 下载我们的预训练模型到 `./models` 文件夹，或使用你自己的预训练模型。
2. 在 `./options.py` 或 `demo.sh` 中修改 `dir_data` 参数为存放数据的位置。
3. 运行 `main.py`，推荐使用脚本文件 `demo.sh`：
```bash
bash demo.sh
```
4. 放大后的图像会保存在 `./experiments/results` 文件夹。