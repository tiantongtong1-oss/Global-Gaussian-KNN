# Global-Gaussian-KNN (CAST v7)

本仓库提取自 [CAST 实验分支](https://github.com/tiantongtong1-oss/CAST/tree/experiment/source-global-gaussian-knn-reliability-v7)，源提交 `a56bfcb3cbf600a52b283e424406cf0d40905b48`。
训练代码和相关测试原样保留，新增本说明、依赖配置与运行脚本；不依赖 CAST 仓库。

## 实验内容

RAF-DB → FER2013 的跨域表情识别，默认 MobileNetV2，包含 EMA teacher、双弱视图、自适应类别阈值、prototype consistency 和 Global Gaussian KNN rescue。
源域真实标签用于估计类别中心，所有类别共享一个各向同性方差；方差的 EMA momentum 默认从 0.70 增长至 0.95（30 次刷新）。候选样本及邻居须满足源域高斯区域约束，邻居还须满足同类别和置信度要求，稀疏邻域受惩罚。

默认 K=20，distribution mass=0.95，score/density threshold=0.5，sparse penalty=0.5；第 3 个目标训练 epoch 起启用 KNN rescue，每 epoch 刷新记忆库。最终接收逻辑以 `combine_rescue_masks` 为准。

## 环境

使用 Python 3.10 与 NVIDIA CUDA GPU。训练代码直接调用 `.cuda()`，目前不能进行 CPU 训练。
以下是为本次迁移整理的固定依赖环境，不代表原实验机器的历史环境；本次迁移环境没有 PyTorch/GPU，未执行数值测试或完整训练。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

PyTorch/torchvision 必须成对安装，CUDA wheel 与驱动需匹配。`numpy==1.23.5` 保留原增强代码使用的 `np.int`；升级 NumPy 前需要处理该兼容性问题。首次构建模型会下载 torchvision 的 ImageNet 预训练权重，离线运行需预先缓存权重。

## 数据准备

数据集、ImageNet 权重和实验 checkpoint 不在源分支中，未包含在此仓库。自行获取 RAF-DB、FER2013，并使用下列结构：

- RAF-DB 根目录：`EmoLabel/list_patition_label.txt` 和 `Image/aligned/`。
- RAF 标签文件每行是原始文件名与 1–7 标签；图片文件名例如 `train_00001_aligned.jpg`。
- FER2013 根目录：`train/0/xxx.jpg`、`val/0/xxx.jpg`、`test/0/xxx.jpg`，各 split 下按 0–6 类别建目录。`val` 也可命名为 `validation`。loader 只查找 `.jpg`。
- FER 原类别顺序：0 angry、1 disgust、2 fear、3 happy、4 sad、5 surprise、6 neutral。代码自动映射为 CAST 顺序：surprise、fear、disgust、happy、sad、angry、neutral，不要预先重复映射。
- 保持训练、验证和测试划分一致；不能把测试集复制为验证集。目标训练标签不参与伪标签筛选，目标验证标签用于选择 checkpoint，测试集用于最终评估。

## 从头运行

```bash
bash scripts/run_v7.sh /absolute/path/to/raf-basic /absolute/path/to/fer2013
```

脚本运行 30 epoch 源域预训练 + 30 epoch 目标域自训练，使用分支默认超参数，固定原代码随机种子 1314。可追加参数覆盖默认值：

```bash
bash scripts/run_v7.sh /data/raf-basic /data/fer2013 --workers 4 --backbone resnet18
python train.py --help
```

必须显式传入数据路径，因为原始 `train.py` 中的默认路径来自作者机器。batch size 在训练入口按 backbone 固定，默认 MobileNetV2 为 128；显存不足需调整 `run_training()` 中的 batch size。

## 已有源域模型初始化

```bash
bash scripts/run_v7.sh /data/raf-basic /data/fer2013 \
  --checkpoint /path/to/source.pth --pre_epochs 0 --epochs 30
```

checkpoint 必须含 `model` 字段且匹配 backbone。此参数只初始化模型权重，不恢复完整目标训练状态；目标记忆库会重建。`--pre_epochs 0` 必须提供 checkpoint。

## 输出与验证

- 运行日志：`logs/v7_<timestamp>.log`。
- 权重：`models/rafdb_fer/*_source_global_gaussian_knn_v7_source_best.pth` 与 `*_target_best.pth`。
- 训练按目标验证集准确率选取最佳模型，结束后输出 `final target test accuracy`。
- `tests/test_global_gaussian_knn_reliability.py` 验证共享方差、momentum、分布约束及稀疏惩罚；其他测试覆盖 KNN、数据 ID、训练接口和 checkpoint。

迁移执行了 Python 语法、脚本语法和源文件一致性检查；没有数据集、源实验完整运行日志或权重，无法核验原实验准确率，也不能保证不同硬件/软件环境逐位一致。

## 文件与来源

`train.py` 串联训练流程；`Networks.py` 提供 backbone 与 CAST 损失；`dataset.py`、`image_utils.py`、`randaugment.py` 提供数据加载和增强；`ema_utils.py`、`prototype_utils.py` 提供 EMA 和原型；`global_gaussian_knn_reliability.py` 是 v7 核心。`knn_reliability.py` 保留供源分支相关回归测试使用。

源分支历史日志属于多个其他实验，未作为 v7 结果复制；未被训练或测试引用的 `energy_utils.py` 不属于本次运行依赖。源 README 中指向不存在文档的链接已用本复现说明替换。源分支未提供 LICENSE，本迁移未新增许可声明；代码来源与增强模块内的原始来源注释均保留。
