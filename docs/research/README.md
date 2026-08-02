# 膝关节超声患者级诊断研究主索引

## 当前状态

- 研究ID：`knee_patient_multimodal_v1_20260724`
- 数据版本：`62ecb01c4d77ec0012704611ecc8d18ef51ebb4e0ea744fb3896948829f0b675`
- 开发队列：967名患者、4,543张图像、六类主要诊断
- 标签隐藏时间外队列：225名患者、1,819张图像，尚未用于任何模型选择
- 当前阶段：C0～C4完成、C3临床参考已冻结；按ADR-008以E3均值特征池化窄范围重开A2，图像融合仍阻塞
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
| 临床输入与实验预登记 | [C0～C4临床输入决策](../project/C0_C4临床输入决策与实验预登记_2026-08-02.md) | 767人异常五分类、字段角色、缺失审计和结果未知时冻结的门槛 |
| 临床五折结果 | [C0～C4临床五折基线与缺失偏差结果](../project/C0_C4临床五折基线与缺失偏差结果_2026-08-02.md) | C3数值基线、C2/C4缺失流程偏差和每类结果 |
| E3实验预登记 | [E3均值特征池化实验预登记](../project/E3_均值特征池化实验预登记_2026-08-02.md) | 用无注意力的患者级等权特征池化检验E2收益来源，冻结门槛与停止规则 |

## 阶段结果

| 阶段 | 状态 | 报告 | 机器产物 |
|---|---|---|---|
| 检查点0 数据可训练 | 完成 | `workspace/.../reports/checkpoint0.md` | registry与数据指纹 |
| E0整图五折 | 完成，阴性/仅风险对照 | [E0/E1五折基线结果](../project/E0_E1整图与ROI五折基线结果_2026-08-02.md) | `reports/oof/E0_oof.csv` |
| E1 ROI五折 | 完成，未达到类别安全门槛 | [E0/E1五折基线结果](../project/E0_E1整图与ROI五折基线结果_2026-08-02.md) | `reports/oof/E1_oof.csv` |
| 非医学代理审计 | 完成，未通过 | [非医学采集与导出代理审计](../project/非医学采集与导出代理审计_2026-08-02.md) | `reports/proxy_audit/*.json` |
| E1-S ROI几何敏感性 | 完成，未通过门槛并停止 | [E1-S五折结果](../project/E1S_ROI几何敏感性消融结果_2026-08-02.md) | `reports/oof/E1S_oof.csv` |
| 检查点A1 输入选择 | 完成 | [ADR-004](../decisions/ADR-004-排除整图主模型并开展ROI几何敏感性消融.md)与实施计划 | 固定人工确认ROI、等比例letterbox，不加边距 |
| E2 Attention MIL | 完成，注意力门槛失败 | [E2五折结果](../project/E2_门控注意力MIL五折结果_2026-08-02.md) | macro-F1 0.3663、比E1高0.1508；pooled塌缩率0.6225 |
| E2-R 单次熵正则补救 | 完成，类别安全门槛失败并停止 | [E2-R五折结果](../project/E2R_注意力熵正则补救五折结果_2026-08-02.md) | macro-F1 0.3623、pooled塌缩率0.3680；痛风/损伤F1下降超限 |
| 检查点A2 图像主模型 | 完成，无合格候选 | [ADR-006](../decisions/ADR-006-E2R未通过类别安全门槛并关闭A2.md) | E1保守参考、E2性能敏感性、E2-R阴性补救；均非临床冻结模型 |
| 临床单模态 C0～C4 | 完成；C3冻结，C2/C4仅作偏差审计 | [C0～C4五折结果](../project/C0_C4临床五折基线与缺失偏差结果_2026-08-02.md) | C3 macro-F1 0.7185；C2 0.5146、C4 0.8050揭示强缺失流程信号 |
| E3 均值特征池化 | 资源pilot通过，待正式五折 | [E3实验预登记](../project/E3_均值特征池化实验预登记_2026-08-02.md) | 5 epochs约6.24分钟、峰值显存1.35GB、外层test未迭代 |
| 图像+Excel融合 | 阻塞 | 待新图像候选的独立方法决策 | 不因融合需求放宽A2门槛 |
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
| [ADR-005](../decisions/ADR-005-E2性能门槛通过但注意力塌缩后启动单次熵正则补救.md) | Accepted | E2不直接晋级；只运行一次固定熵正则E2-R，暂缓骨干挑战 |
| [ADR-006](../decisions/ADR-006-E2R未通过类别安全门槛并关闭A2.md) | Accepted | E2-R类别安全失败；关闭A2且不伪选主模型，先继续临床单模态 |
| [ADR-007](../decisions/ADR-007-冻结C3临床数值基线并限制缺失模式用途.md) | Accepted | 冻结C3；C2/C4只作工作流偏差与敏感性，不启动额外临床模型搜索 |
| [ADR-008](../decisions/ADR-008-以E3均值特征池化重开图像候选评估.md) | Accepted | 用患者级等权embedding平均区分患者目标收益与注意力收益，窄范围重开A2 |

## 实验追踪

- 本地MLflow数据库：`workspace/experiments/active/exp_2026-07_patient_multimodal_v1/tracking/mlflow.db`
- MLflow实验：`patient-primary-diagnosis`
- Markdown解释：本索引、阶段报告和[实验结构化登记](experiment_ledger.yaml)
- OOF与评价：`workspace/experiments/active/exp_2026-07_patient_multimodal_v1/reports/`
- checkpoint：`workspace/experiments/active/exp_2026-07_patient_multimodal_v1/artifacts/`

MLflow用于逐epoch机器记录；Markdown负责研究问题、预设门槛、阴性结果、偏差解释和停止决定。任何一边缺失都不能算阶段完成。

## 按日期研究日志

- [2026-08-02](logs/2026-08-02.md)：E0/E1/E1-S/E2/E2-R和C0～C4五折、代理/缺失偏差审计、A1/A2及临床输入决策、全部失败路线和持续文档验收。

## 隐私边界

- Git、MLflow、Markdown和公开评价文件只使用伪匿名 `person_key`/`image_key`；
- 不登记姓名、原始患者目录、图像文件名、完整原始路径或日期；
- 私有身份关联仅保存在被Git忽略的registry `private/`；
- 文档中的产物路径只到研究目录和伪匿名文件，不列原始数据路径。

## 最终报告连续生成要求

最终报告必须能够从本索引、日期日志、实验登记和阶段报告连续追溯：纳入排除、数据版本、实验演进、阴性/停止路线、非医学捷径、每类结果、临床融合、一次性盲测和局限性。不能只选取最佳实验。
