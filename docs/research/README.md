# 膝关节超声患者级诊断研究主索引

## 当前状态

- 研究ID：`knee_patient_multimodal_v1_20260724`
- 数据版本：`62ecb01c4d77ec0012704611ecc8d18ef51ebb4e0ea744fb3896948829f0b675`
- 开发队列：967名患者、4,543张图像、六类主要诊断
- 标签隐藏时间外队列：225名患者、1,819张图像，尚未用于任何模型选择
- 当前阶段：检查点A1已关闭；E1-S五折阴性，开始E2 Attention MIL实现与资源试运行
- 研究用途：内部模型开发，不用于当前临床独立决策

本索引是研究文档的唯一总入口。机器产物保存在本地实验目录，解释、决策和阴性结果保存在Git文档；两者必须互相引用。

## 数据和方法入口

| 内容 | 文档/产物 | 说明 |
|---|---|---|
| 客户训练要求 | 本地受保护需求文件 | 原始需求只读保留，研究报告不展开原始路径 |
| 研究方案 | [新一轮患者级多模态诊断研究方案](../project/新一轮患者级多模态诊断研究方案_2026-07-24.md) | 患者级六分类、图像和临床融合总策略 |
| 实施计划 | [研究实施计划 v1.0](../project/新一轮患者级多模态诊断研究实施计划_v1.0_2026-07-24.md) | 阶段、门槛、资源限制与完成状态 |
| 数据冻结报告 | `workspace/experiments/active/exp_2026-07_patient_multimodal_v1/reports/checkpoint0.md` | 纳入排除、类别、ROI、身份和五折审计 |
| 数据登记目录 | `workspace/data/registry/exp_2026-07_patient_multimodal_v1/` | 患者、图像、固定外层/内层折；私有关联表不进入Git |
| 实验结构化登记 | [experiment_ledger.yaml](experiment_ledger.yaml) | 配置、代码、运行、OOF和结论的机器可读登记 |
| 文档验收规范 | [研究文档与产物验收规范](研究文档与产物验收规范.md) | 每阶段和每个正式实验的强制字段 |

## 阶段结果

| 阶段 | 状态 | 报告 | 机器产物 |
|---|---|---|---|
| 检查点0 数据可训练 | 完成 | `workspace/.../reports/checkpoint0.md` | registry与数据指纹 |
| E0整图五折 | 完成，阴性/仅风险对照 | [E0/E1五折基线结果](../project/E0_E1整图与ROI五折基线结果_2026-08-02.md) | `reports/oof/E0_oof.csv` |
| E1 ROI五折 | 完成，未达到类别安全门槛 | [E0/E1五折基线结果](../project/E0_E1整图与ROI五折基线结果_2026-08-02.md) | `reports/oof/E1_oof.csv` |
| 非医学代理审计 | 完成，未通过 | [非医学采集与导出代理审计](../project/非医学采集与导出代理审计_2026-08-02.md) | `reports/proxy_audit/*.json` |
| E1-S ROI几何敏感性 | 完成，未通过门槛并停止 | [E1-S五折结果](../project/E1S_ROI几何敏感性消融结果_2026-08-02.md) | `reports/oof/E1S_oof.csv` |
| 检查点A1 输入选择 | 完成 | [ADR-004](../decisions/ADR-004-排除整图主模型并开展ROI几何敏感性消融.md)与实施计划 | 固定人工确认ROI、等比例letterbox，不加边距 |
| E2 Attention MIL | 实现完成，待资源试运行 | 实施计划与当日日志 | 配置 `configs/research/e2_roi_gated_attention_b2.yaml` |
| 图像+Excel融合 | 未开始 | 待图像主模型冻结 | 无 |
| 2026标签隐藏盲测 | 隔离 | 最终模型冻结后一次性执行 | 尚无预测 |

其中 `workspace/...` 均指：

```text
workspace/experiments/active/exp_2026-07_patient_multimodal_v1/
```

## 架构与研究决策

| ADR | 状态 | 决策 |
|---|---|---|
| [ADR-001](../decisions/ADR-001-采用患者级六类主要诊断与分层多模态融合.md) | Accepted | 患者级六类主要诊断与分层多模态融合 |
| [ADR-002](../decisions/ADR-002-固定临床特征白名单与标注规范化前置条件.md) | Accepted | 临床字段白名单和28类疾病无关标注 |
| [ADR-003](../decisions/ADR-003-限制单模型训练时间并保护本机CPU资源.md) | Accepted | 单run时间、GPU、CPU和内存保护 |
| [ADR-004](../decisions/ADR-004-排除整图主模型并开展ROI几何敏感性消融.md) | Accepted | 排除整图与ROI边距；E1-S失败后冻结等比例letterbox ROI |

## 实验追踪

- 本地MLflow数据库：`workspace/experiments/active/exp_2026-07_patient_multimodal_v1/tracking/mlflow.db`
- MLflow实验：`patient-primary-diagnosis`
- Markdown解释：本索引、阶段报告和[实验结构化登记](experiment_ledger.yaml)
- OOF与评价：`workspace/experiments/active/exp_2026-07_patient_multimodal_v1/reports/`
- checkpoint：`workspace/experiments/active/exp_2026-07_patient_multimodal_v1/artifacts/`

MLflow用于逐epoch机器记录；Markdown负责研究问题、预设门槛、阴性结果、偏差解释和停止决定。任何一边缺失都不能算阶段完成。

## 按日期研究日志

- [2026-08-02](logs/2026-08-02.md)：E0/E1正式五折汇总、非医学代理审计、两次性能原因停止、E1-S五折阴性结果、A1输入冻结与ADR-004接受。

## 隐私边界

- Git、MLflow、Markdown和公开评价文件只使用伪匿名 `person_key`/`image_key`；
- 不登记姓名、原始患者目录、图像文件名、完整原始路径或日期；
- 私有身份关联仅保存在被Git忽略的registry `private/`；
- 文档中的产物路径只到研究目录和伪匿名文件，不列原始数据路径。

## 最终报告连续生成要求

最终报告必须能够从本索引、日期日志、实验登记和阶段报告连续追溯：纳入排除、数据版本、实验演进、阴性/停止路线、非医学捷径、每类结果、临床融合、一次性盲测和局限性。不能只选取最佳实验。
