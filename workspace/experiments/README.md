# Experiments

本目录用于隔离每一次正式实验，避免配置、结果、日志和总结彼此污染。

约定：

- `archive/`：已完成或历史实验归档
- `active/`：当前正在推进的实验
- `templates/`：新实验目录模板

每个实验目录建议包含：

- `README.md`
- `manifest.yaml`
- `notes.md` / `journal.md`
- `configs/`
- `logs/`
- `artifacts/`
- `reports/`

当前实验状态：

- 历史全图基线：`archive/exp_2026-04_fullimage_legacy_baseline`
- ROI-only 分类实验：`active/exp_2026-04_roi_only_cls`

后续新增实验时，不再向根目录写入 `runs/`、`configs/`、`reports/` 一类公共结果目录。
