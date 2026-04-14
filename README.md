# ultrasound_recog

膝关节超声图像 AI 识别项目，当前按“共享平台层 + 实验隔离层”组织。

## 目录结构

```text
ultrasound_recog/
├── annotation_viewer.py
├── assets/
│   └── pretrained/
├── docs/
│   └── project/
├── src/
├── tools/
├── workspace/
│   ├── data/
│   │   ├── raw/
│   │   ├── registry/
│   │   └── shared_derived/
│   ├── experiments/
│   │   ├── archive/
│   │   └── active/
│   └── reports/
├── requirements.txt
└── .gitignore
```

## 当前实验

- 历史全图基线：
  `workspace/experiments/archive/exp_2026-04_fullimage_legacy_baseline`
- 当前 ROI-only 分类实验：
  `workspace/experiments/active/exp_2026-04_roi_only_cls`

每个实验目录独立保存：

- `README.md`
- `manifest.yaml`
- `notes.md` / `journal.md`
- `configs/`
- `logs/`
- `artifacts/`
- `reports/`

## 共享入口

- 标注与 ROI 编辑：
  `python annotation_viewer.py`
- 通用脚本：
  `tools/`

常用脚本：

```text
tools/01_clean_labels.py
tools/02_split_dataset.py
tools/03_convert_coco.py
tools/04_convert_yolo.py
tools/05_convert_nnunet.py
tools/06_convert_medsam.py
tools/07_train_yolo.py
tools/08_train_nnunet.sh
tools/09_train_classifier.py
tools/10_train_medsam.py
tools/11_evaluate.py
tools/12_inference.py
tools/13_demo_app.py
```

## 数据与输出

- 原始数据：
  `workspace/data/raw/膝关节已标注`
- 共享派生数据：
  `workspace/data/shared_derived/`
- 历史全图实验产物：
  `workspace/experiments/archive/exp_2026-04_fullimage_legacy_baseline/artifacts/`

## 文档入口

- 项目级文档与日志：
  `docs/project/`
- 实验说明：
  `workspace/experiments/README.md`
- 实验索引：
  `workspace/reports/experiment_index.md`

## 说明

- `src/common_paths.py` 是路径单一来源。
- 根目录只保留共享代码、主界面入口和平台级目录。
- 根目录 `data/` 目前只残留一个被系统占用的空目录壳，实际工作路径已切换到 `workspace/data/raw/`。
