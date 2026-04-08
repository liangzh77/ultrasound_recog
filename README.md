# ultrasound_recog

膝关节超声图像 AI 识别项目，面向多疾病场景下的超声图像分割与疾病分类。

项目围绕同一批膝关节超声数据，比较了 YOLO、nnU-Net、MedSAM 三条技术路线，并提供数据预处理、训练、评估、推理和 Gradio 演示脚本。

## 项目概览

根据现有研究文档，数据集大致包含：

- 730 名患者
- 6866 张膝关节超声图像
- 3420 张已标注图像
- 7 类疾病
- 80+ 解剖/病灶标注类别

## 技术路线

当前项目主要包含三种方案：

1. `YOLO11-seg + EfficientNet-B0`
   用于实例分割 + 疾病分类
2. `nnU-Net v2 + EfficientNet-B0`
   用于医学影像语义分割 + 疾病分类
3. `MedSAM + 分类头`
   用于基础模型微调与 prompt 分割

## 项目结构

```text
ultrasound_recog/
├── configs/             # 训练配置
├── data/                # 数据集与转换产物
├── runs/                # 训练结果、权重、推理输出
├── scripts/             # 数据处理、训练、评估、推理、演示脚本
├── src/                 # 公共工具函数
├── requirements.txt     # Python 依赖
├── 研究方案.md
├── 研究总结.md
├── 工作日志.md
└── 开发日志.md
```

## 主要脚本

`scripts/` 目录已覆盖完整流程：

```text
01_clean_labels.py    标签清洗
02_split_dataset.py   数据集拆分
03_convert_coco.py    转 COCO
04_convert_yolo.py    转 YOLO
05_convert_nnunet.py  转 nnU-Net
06_convert_medsam.py  转 MedSAM
07_train_yolo.py      训练 YOLO
08_train_nnunet.sh    训练 nnU-Net
09_train_classifier.py 训练分类器
10_train_medsam.py    微调 MedSAM
11_evaluate.py        统一评估
12_inference.py       统一推理
13_demo_app.py        Gradio 演示
```

## 环境要求

- Python 3.10+
- 建议使用 GPU 环境运行训练与推理
- 部分方案依赖 PyTorch / CUDA

核心依赖见 [`requirements.txt`](./requirements.txt)：

- `ultralytics`
- `nnunetv2`
- `torch`
- `torchvision`
- `timm`
- `monai`
- `segment-anything`
- `opencv-python`
- `gradio`

## 安装

建议先创建虚拟环境，再安装依赖：

```bash
pip install -r requirements.txt
```

如果需要运行 MedSAM 方案，还需确认以下权重文件可用：

- `medsam_vit_b.pth`
- `yolo11m-seg.pt`
- `yolo26n.pt`

## 训练配置

当前 `configs/` 目录下包含：

- [`configs/yolo_seg.yaml`](./configs/yolo_seg.yaml)
- [`configs/yolo_seg_merged.yaml`](./configs/yolo_seg_merged.yaml)

从研究总结看，项目对原始多类别标签做过合并策略实验，`merged` 配置通常对应合并类别后的训练方案。

## 推理

统一推理脚本：

```bash
python scripts/12_inference.py --method yolo --image path/to/image.jpg
python scripts/12_inference.py --method nnunet --image path/to/image.jpg
python scripts/12_inference.py --method medsam --image path/to/image.jpg --sam-weights medsam_vit_b.pth
```

`12_inference.py` 支持三种后端：

- `yolo`
- `nnunet`
- `medsam`

其中：

- YOLO 路径会输出解剖结构分割结果，并调用 EfficientNet-B0 做疾病分类
- nnU-Net 当前主要通过命令行推理分割，脚本中保留分类汇总说明
- MedSAM 需要额外提供 SAM 预训练权重

## 演示界面

可启动 Gradio Demo：

```bash
python scripts/13_demo_app.py
```

可选参数：

```bash
python scripts/13_demo_app.py --share
python scripts/13_demo_app.py --port 7861
```

该演示界面主要集成：

- YOLO11-seg 分割结果可视化
- EfficientNet-B0 疾病分类
- 中文标签绘制

## 输出目录

- `data/`：原始数据、拆分结果、各模型格式转换数据
- `runs/`：训练权重、日志、推理输出

如果运行脚本后没有结果，优先检查：

- 训练权重是否存在于 `runs/` 下
- 输入路径是否正确
- GPU / CUDA 环境是否可用
- MedSAM 权重路径是否正确

## 文档索引

- [研究方案.md](./研究方案.md): 项目研究背景、方法设计与前沿综述
- [研究总结.md](./研究总结.md): 三条技术路线的实验结果总结
- [工作日志.md](./工作日志.md): 过程记录
- [开发日志.md](./开发日志.md): 开发记录

## 说明

这是一个偏研究与实验性质的工程，目录中已包含权重文件、日志文件和实验产物。实际复现时，建议先从 `scripts/12_inference.py` 和 `scripts/13_demo_app.py` 验证环境，再决定是否完整重跑训练流程。
