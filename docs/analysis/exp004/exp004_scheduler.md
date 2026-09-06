# exp004 学习率调度实验分析

> 本文记录 exp004 的真实训练结果。实验于 2026-09-06 在 AutoDL RTX 5090 完成。

## 实验目的

验证在 exp003 配方基础上加入 warmup + cosine 学习率调度，能否改善恒定学习率训练末期仍在上升的问题。

## 配置

| 项目 | 值 |
|---|---|
| 模型 | SegFormer MIT-B0 |
| 输入尺寸 | 768 x 768 |
| batch size | 2 |
| 初始学习率 | 6e-5 |
| 优化器 | AdamW，weight decay 0.01 |
| loss | CrossEntropy，Ignore=0 |
| 增强 | 水平翻转 |
| 训练轮数 | 60 |
| 学习率策略 | warmup 2 epoch + cosine decay |
| 数据划分 | 固定 train.csv / val.csv，5597 / 1399 |
| 设备 | AutoDL RTX 5090 |
| 训练耗时 | 204.57 分钟 |

除学习率策略外，其余主要配置与 exp003 保持一致。

## 结果

| 指标 | exp003 | exp004 | 变化 |
|---|---:|---:|---:|
| 最佳 epoch | 60 | **56** | -4 |
| 最佳验证 mIoU | 0.7791120845 | **0.7859696952** | **+0.0068576107** |
| 最后一轮 mIoU | 0.7791120845 | 0.7854496997 | +0.0063376152 |
| 预测 Ignore 比例 | 0 | 0 | 无变化 |

exp004 最佳验证 mIoU 达到 `0.7859696952`，超过路线图设定的 `0.782` 保留标准，因此保留为新的本地验证基准。

## 每类 IoU 对比

| ID | 类别 | exp003 | exp004 | 变化 |
|---:|---|---:|---:|---:|
| 1 | Background | 0.715195 | 0.720741 | +0.005545 |
| 2 | Building | 0.844602 | 0.848499 | +0.003897 |
| 3 | Road | 0.818184 | 0.821128 | +0.002944 |
| 4 | Water | 0.878559 | 0.879135 | +0.000577 |
| 5 | Barren | 0.558752 | 0.570985 | **+0.012233** |
| 6 | Vegetation | 0.882134 | 0.888825 | +0.006690 |
| 7 | Agricultural | 0.773922 | 0.786089 | **+0.012167** |
| 8 | Vehicle | 0.761548 | 0.772355 | **+0.010807** |

8 个有效类别全部提升，没有出现以牺牲其他类别换取总 mIoU 上升的情况。Barren、Agricultural 和 Vehicle 的提升最明显，说明 scheduler 对 exp003 尚未完全稳定的后半程优化有实际帮助。

## 学习率和收敛判断

- epoch 1 学习率为 `6e-6`；
- epoch 2 学习率为 `3.3e-5`；
- epoch 56 学习率约为 `1.09e-6`，达到最佳验证结果；
- epoch 60 学习率约为 `4.40e-8`，末轮 mIoU 略低于 epoch 56。

与 exp003 的最佳轮次为最后一轮相比，exp004 的最佳轮次提前到 56，说明 cosine 衰减提供了更合理的收敛终点，但 60 epoch 已基本达到尾部。当前没有证据说明必须继续延长到 80 或 100 epoch。

## 当前结论

1. exp004 scheduler 成功，保留为新的本地验证基准。
2. 下一轮不再重复无 scheduler 长训；后续实验应从 exp004 的最佳配置出发。
3. 线上 test_1 尚未验证 exp004 的增益，建议使用 exp004 epoch 56 的 `best_model` 按相同 768 resize 管线提交一次，作为 scheduler 的外部校准。
4. 在提交 exp004 后，优先推进多尺度和受控域增强；不建议同时加入 Dice、类别权重和更大骨干。
5. TTA 可以用 exp004 best checkpoint 做一次零训练评估；若固定验证集增益低于约 0.002，则继续使用原始 768 resize。

## 结果文件

- 原始结果：`outputs/results/exp004_scheduler/`
- 配置：`run_config.json`
- 训练记录：`history.json`、`history.jsonl`
- 最优指标：`best_metrics.json`
- 训练摘要：`summary.json`
- AutoDL 远端模型：`/root/autodl-tmp/runs/exp004_scheduler/best_model`

## S2 提交工件

已使用 epoch 56 `best_model` 和固定 768 resize 推理生成 test_1 预测。提交包信息：

- 文件：`outputs/submissions/exp004_scheduler_resize768.zip`
- 大小：`5,183,029` 字节
- SHA-256：`96525B2F171BA60392ECB764E3E9F7BE24B180019244EAB69E9A27DDEA67D522`
- 条目：500 个根目录 PNG，0 个嵌套路径，0 个非 PNG 条目
- 图像校验：全部为 1024 x 1024、单通道 `L`，标签值属于 0-8

线上分数尚待在比赛平台提交后补充。
