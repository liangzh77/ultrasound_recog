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
- 当前 ROI 多边形自动标注实验：
  `workspace/experiments/active/exp_2026-04_roi_poly_seg`
- 当前患者级多模态诊断研究：
  `workspace/experiments/active/exp_2026-07_patient_multimodal_v1`
  — 已完成数据冻结、28 类规范化、患者 manifest、固定五折、E0/E1正式五折和非医学代理审计；审计发现强采集/导出捷径，当前执行E1-S几何敏感性消融后关闭检查点A1。

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
tools/14_train_roi_poly_seg.py
tools/15_visualize_roi_poly_seg.py
tools/16_demo_roi_poly_seg.py
tools/16_demo_roi_poly_seg_tk.py
tools/17_evaluate_roi_poly_seg.py
tools/19_check_research_environment.py
tools/20_fingerprint_research_sources.py
tools/21_normalize_research_annotations.py
tools/22_build_research_manifest.py
tools/23_build_patient_folds.py
tools/24_visualize_research_inputs.py
tools/25_evaluate_patient_oof.py
tools/26_train_patient_image_mean.py
```

## 数据与输出

- 原始数据：
  `workspace/data/raw/膝关节已标注`
- 共享派生数据：
  `workspace/data/shared_derived/`
- 历史全图实验产物：
  `workspace/experiments/archive/exp_2026-04_fullimage_legacy_baseline/artifacts/`

## 文档入口

- 患者级诊断研究主索引：
  `docs/research/README.md`
  — 持续索引数据版本、ADR、阶段报告、实验登记、日期日志、MLflow与本地OOF产物；正式实验缺少哈希或阴性结果记录时不得标记完成。
- 项目级文档与日志：
  `docs/project/`
- 标注工具定位与启动说明：
  `docs/project/标注工具定位与启动说明.md`
  — 说明像素级标注工具 ISAT 的实际位置和启动方法，区分 ISAT、项目 ROI 编辑器与自动标注演示，解决“标注程序在哪里、应该启动哪个程序”的问题。
- 原始标注数据同步记录：
  `docs/project/原始标注数据同步记录_2026-07-24.md`
  — 记录患者目录自动匹配与重命名、新增数据和 Excel 的同步结果、完整性校验，以及重新生成清洗数据和训练集划分时的注意事项。
- 像素区域标签规范化记录：
  `docs/project/像素区域标签规范化记录_2026-07-24.md`
  — 完整审计 88 个原始区域类别，将疾病前缀和已确认的错别字统一规范化为 28 个疾病无关类别，并列出全部映射、对象数和仍需临床确认的近似名称。
- 临床 Excel 更新替换记录：
  `docs/project/临床Excel更新替换记录_2026-07-24.md`
  — 记录客户更新的 5 份临床工作簿与疾病目录的对应关系、新旧内容差异、旧版备份位置和替换后的完整性校验；其中 RA 修正 2 个 RF 值，损伤表修正化验列错位。
- Excel 训练特征与标签泄漏审计：
  `docs/project/Excel训练特征与标签泄漏审计_2026-07-24.md`
  — 审计诊断、类别编码、病程、日期和化验缺失模式造成的标签泄漏，给出临床特征白名单、分层融合策略，并指出当前 28 类区域规范化尚未真正更新训练缓存。
- 新一轮患者级多模态诊断研究方案：
  `docs/project/新一轮患者级多模态诊断研究方案_2026-07-24.md`
  — 基于新增数据和客户补充要求，确定患者级六类主要诊断、ROI 动态输入、Attention MIL、疾病无关病变识别、正常组无 Excel 时的分层融合、五折 OOF 验证和 2026 隐藏标签盲测方案，并给出可验收的实施任务。
- 新一轮患者级多模态诊断研究实施计划：
  `docs/project/新一轮患者级多模态诊断研究实施计划_v1.0_2026-07-24.md`
  — 将研究方案落实为数据冻结、外层五折/内层早停、E0～F2 实验矩阵、默认训练参数、统计门槛、RTX 3080 与磁盘预算、可执行任务和阶段停止规则。
- 患者级多模态诊断开发阶段综合研究报告：
  `docs/project/患者级多模态诊断开发阶段综合研究报告_2026-08-03.md`
  — 连续汇总967人/4,543图的纳入排除、E0～E3/G0/C0～C4/S1/H0演进、每类结果、失败路线、采集与缺失偏差、2026盲测未执行状态和局限性；当前没有合格冻结诊断模型。
- 研究剩余任务与停止状态检查点：
  `docs/project/研究剩余任务与停止状态检查点_2026-08-03.md`
  — 区分已完成、已停止和等待外部输入的任务，防止误把E2-C、F0～F2或Z1/Z2当成尚可继续调参的待办。
- 研究恢复条件复核检查点：
  `docs/project/研究恢复条件复核检查点_2026-08-03.md`
  — 逐文件复核16,809个冻结输入及S1a 8+8、双签署状态；确认当前没有新增数据或临床回表可合法重启训练。
- S1a 临床确认包准备结果：
  `docs/project/S1a临床确认包准备结果_2026-08-03.md`
  — 索引可直接交给医生填写的五页确认工作簿、数据/标注/Git版本、文件哈希和冻结前机器门；八项医学定义和八项复核门槛未书面确认前，正式复核与训练保持关闭。
- S1a 临床确认机器门与工作簿对账：
  `docs/project/S1a临床确认机器门结果_2026-08-03.md`、`docs/project/S1a临床确认工作簿对账检查点_2026-08-03.md`
  — 说明临床回表如何绑定哈希、只读逐项核对Excel/YAML、保护姓名并在任何差异时失败关闭；当前仍等待真实临床回表。
- S1a 正式语义复核录入工程：
  `docs/project/S1a正式语义复核录入工程准备结果_2026-08-03.md`
  — 记录两位复核者独立盲法录入、原子保存、配置/队列/Git硬门和受控合并；并说明为何将多边形Dice/IoU拆到S1b单独预登记。当前Draft启动会在写文件前拒绝。
- S1a 盲法分歧裁决工程：
  `docs/project/S1a盲法分歧裁决工程准备结果_2026-08-03.md`
  — 记录只裁决分歧字段、禁止覆盖一致字段、存在状态/子型一致性门、裁决清单和最终统计前完整裁决硬门。当前Draft启动会在写文件前拒绝。
- S1a 正式复核执行操作手册：
  `docs/project/S1a正式复核执行操作手册_2026-08-03.md`
  — 说明临床回表后如何按严格门、冻结配置、双人独立录入、合并、盲法裁决、最终一致性统计和文档登记的顺序执行。
- G0 患者级正常/异常图像门控预登记：
  `docs/project/G0患者级正常异常图像门控实验预登记_2026-08-03.md`
  — 冻结单一二分类患者MIL候选及九项门槛；正式五折已完成，因折间稳定性和ECE失败而停止，未进入融合或2026盲测。
- E0/E1 整图与 ROI 五折基线结果：
  `docs/project/E0_E1整图与ROI五折基线结果_2026-08-02.md`
  — 记录 967 名患者正式五折 OOF、配对 bootstrap、每类灵敏度和 ROI 门槛判断，说明为何检查点 A1 尚不能直接冻结 ROI。
- 非医学采集与导出代理审计：
  `docs/project/非医学采集与导出代理审计_2026-08-02.md`
  — 证明整图边框、ROI外背景、采图数量和ROI几何能够显著预测疾病，排除整图主模型并预注册一次E1-S几何敏感性消融。
- 架构决策记录：
  `docs/decisions/ADR-001-采用患者级六类主要诊断与分层多模态融合.md`
  — 说明为何排除滑膜囊肿主要诊断、为何不能直接做六类图像/Excel 融合，以及为何采用患者多图和软融合架构。
- 正常/异常图像门控决策：
  `docs/decisions/ADR-013-以独立正常异常图像门控重开分层融合.md`
  — 接受独立 G0 二分类任务以补齐原分层架构；不改变 E1～E3 阴性结论，只有 G0 全部门槛通过后才允许另行预登记融合。
- 临床特征与标注规范化决策：
  `docs/decisions/ADR-002-固定临床特征白名单与标注规范化前置条件.md`
  — 固定后续训练的临床字段白名单和正常组分层融合约束，把重新生成 28 类疾病无关标注设为训练前置条件，并记录随机化验项遮蔽作为推荐的稳健性实验。
- 本机训练资源与时限决策：
  `docs/decisions/ADR-003-限制单模型训练时间并保护本机CPU资源.md`
  — 固定 RTX 3080 10 GB 下的主模型、CPU/内存/显存保护、单 fold 资源试运行、10 小时目标、11.5 小时软截止和 23.5 小时硬截止。
- 实验说明：
  `workspace/experiments/README.md`
- 实验索引：
  `workspace/reports/experiment_index.md`

## 说明

- `src/common_paths.py` 是路径单一来源。
- 根目录只保留共享代码、主界面入口和平台级目录。
- 根目录 `data/` 目前只残留一个被系统占用的空目录壳，实际工作路径已切换到 `workspace/data/raw/`。
