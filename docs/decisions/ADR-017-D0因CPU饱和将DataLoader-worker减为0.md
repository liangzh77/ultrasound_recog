# ADR-017：D0因CPU饱和将DataLoader worker减为0

- 状态：Accepted
- 日期：2026-08-03
- 范围：D0异常五分类ROI患者MIL的本机资源配置

## 背景

D0初始资源pilot在5 epoch内通过时间、显存、数值和outer-test隔离门，但未把系统总CPU持续峰值作为独立验收项。首次正式fold 0运行时，`num_workers=2`分别为训练和验证DataLoader生成工作进程，完成第1个epoch后同时存在四个加载进程，系统总CPU持续接近100%。这与用户要求的“不要把CPU卡死”直接冲突。

该运行在outer test未迭代时主动终止，MLflow parent/child均标记`FAILED`，不解释验证性能。

## 决策

将D0的`data.num_workers`从2减为0，然后重新执行同样的fold 0、5-epoch资源pilot。新pilot除原门槛外，增加“系统总CPU不持续饱和”验收。只有新pilot通过后才能重启正式五折。

不改变任何研究假设或模型选择项：患者队列、ROI几何、EfficientNet-B2、gated attention、初始权重、dropout、优化器、学习率、有效batch、fold/seed、epoch/早停、KL系数、性能/类别安全/校准/注意力门槛全部不变。

## 理由

`num_workers=0`使图像读取和增强在训练主进程串行执行，会降低GPU利用率并可能延长单epoch，但pilot的时间余量足以在12小时目标内完成正式单折。这是保护用户本机响应性的最小资源变更，不会引入新候选搜索。

## 后果

- 旧配置SHA-256 `6c7da2c0e83924bcb20ac75c4c54cdf32a5b57dee017ab21ce9de7b4f4683e0d`仍与旧pilot和失败尝试绑定；
- 修订配置SHA-256为`1f6908242864b404a360359a4cf8a2c1a801b86125997d39e1b2deb78f999f58`；
- 失败尝试、部分checkpoint和MLflow失败状态永久保留；
- 禁止用失败尝试的checkpoint恢复新pilot或正式fold。
