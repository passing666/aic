# exp003 线上结果后的冲刺规划：从 0.608668 向 0.72

> 制定日期：2026-09-06
>
> 本文综合 `docs/analysis/exp003/`、`docs/suggestions/exp003/`、`docs/suggestions/target_072_strategy_glm.md`、`docs/suggestions/roadmap_to_0921_glm.md` 以及当前工作区的实验记录。
>
> 本文是决策规划，不是实验事实。每次训练完成后，真实结果以对应实验目录和事实分析文档为准。

## 1. 当前结论

### 1.1 已经验证的事实

| 项目 | exp002 | exp003 | 变化 |
|---|---:|---:|---:|
| 固定验证集 mIoU | 0.762585 | 0.779112 | +0.016527 |
| test_1 官方 mIoU | 0.591572 | **0.608668** | **+0.017096** |
| val - test | 0.171013 | 0.170444 | -0.000569 |

两次提交使用相同的 MIT-B0、768 输入、推理方式、数据划分和评价链路，主要变量是训练轮数从 30 增加到 60。因此可以得到一个可靠的局部结论：

- exp003 的本地提升几乎按 1:1 转化为线上提升；
- 继续改善训练配方有现实价值，不是只优化验证集；
- 当前线上成绩距离约 0.72 仍差 `0.111332`，不能靠再增加几十个 epoch 解决；
- `val - test` 在两次提交中接近，但样本数只有两次，不能当成所有未来测试集的固定定律。

### 1.2 对 0.72 目标的正确理解

`0.72` 应作为冲刺目标和资源分配目标，而不是保证值。达到它需要同时改善两部分：

1. 提高模型在固定验证集上的真实能力；
2. 减少训练/验证分布与官方测试分布之间的泛化损失。

按照当前 test_1 的经验差距估算，若差距仍接近 `0.17`，需要验证集达到约 `0.89` 才能线上达到 `0.72`，这对当前 MIT-B0 配方并不现实。因此必须测试能够改变域泛化的措施，而不是只堆本地 mIoU。

更合理的阶段性目标是：

| 阶段 | 线上目标 | 含义 |
|---|---:|---|
| 保底 | > 0.608668 | 新实验不能轻易回退线上基准 |
| 第一阶段 | 0.63 左右 | scheduler、增强、推理和训练工程稳定兑现 |
| 第二阶段 | 0.66–0.69 | 多尺度、强增强、EMA 或更大骨干产生有效收益 |
| 冲刺目标 | 约 0.72 | 需要模型能力和跨域泛化同时明显改善，存在不确定性 |

## 2. 总体策略

采用“两条线并行、训练实验串行”的方式：

- **训练主线**：严格控制变量，先验证 scheduler，再验证吞吐配方、增强、loss 和骨干；
- **泛化与工程线**：扩展 stressed validation、做推理策略评估、准备 Docker 和复赛提交 SOP；
- **排行榜提交**：test_1 没有提交次数限制，但仍不把它当作训练标签。只在关键候选配方完成后提交，用于验证方向和估计泛化，不能根据单次分数反复微调到 test_1。

核心原则：

1. 固定 `train.csv`、`val.csv`、随机种子和评价口径；
2. 每个实验目录独立保存，不覆盖历史结果；
3. 训练实验至少记录总 mIoU、8 类 IoU、`predicted_ignore_ratio`、最佳 epoch、耗时和完整配置；
4. 低于当前验证基准的实验只在它提供明确诊断价值时保留；
5. 所有外部权重、额外数据和伪标签方案先核对比赛规则，未获确认不使用；
6. 继续遵守单模型约束，不使用多模型 ensemble 或权重平均作为最终提交。

## 3. 官方参考方法与我们自己的方法

### 3.1 官方参考资源不等于必做清单

官方题目页面的参考资源包括 SegFormer、MMSegmentation 和 SAM3，同时在解题思路中提到加权交叉熵、Focal Loss、Dice Loss、颜色抖动、多尺度增强和域适应方向。这些内容的定位是提供起点和思路，不是要求参赛队伍全部实现。

当前项目已经使用 SegFormer MIT-B0，满足官方推荐的开源语义分割基线方向。MMSegmentation 是训练框架替代方案，不是必须叠加到 Hugging Face 训练链路上；在当前阶段迁移框架会增加复现和评价口径风险。SAM3 主要是通用视觉分割/提示式模型，不能因为出现在参考资源中就默认适合本题的 9 类像素级监督任务，也不能与主模型组成最终 ensemble。

### 3.2 高分需要“可验证的自己的方法”，不需要为了创新而复杂化

我们的特有方法应建立在本题已观察到的具体问题上：

1. 训练集与 test_1 存在明显域差距；
2. Barren 容易被 Vegetation 吞并；
3. Vehicle、Road 的细小目标和细长边界在 768 resize 中损失明显；
4. 测试集类别分布可能与训练集不同；
5. 官方要求单模型、禁止额外数据和多模型集成。

因此更有说服力的创新路线是“面向无人机跨域分布的单模型多尺度鲁棒分割配方”，由可拆解的模块组成：

- 多尺度训练解决目标尺寸变化；
- 受控颜色/通道增强解决传感器和场景域偏移；
- CE + 温和 Dice 或其他少数类机制解决 Barren 吞并；
- TTA/EMA 只作为单模型推理和稳定化选项；
- 固定验证集 + stressed validation + 关键 test_1 校准证明每个模块的收益。

这已经可以形成自己的方法论。创新性来自问题诊断、增强与损失的针对性设计、严格的消融证据和可复现闭环，而不是把多个热门模型或损失函数机械堆在一起。

## 4. 5090 并行执行方案

可以并行，但并行单位应是“互不依赖的实验或诊断”，不是让多个高显存训练进程同时争抢同一张 GPU。5090 空余显存可以用于增大单个实验 batch、加快验证和做 benchmark；不建议默认同时启动两个 768 训练进程。

### 4.1 推荐资源分配

| 队列 | 工作 | 是否占用大显存 | 与训练关系 |
|---|---|---:|---|
| A | exp004 scheduler | 高 | 主训练，优先独占 GPU |
| B | exp003 TTA / stressed validation | 中 | 可与 A 并行，但需错开显存高峰或使用独立 GPU |
| C | batch benchmark | 中到高 | 在训练空档运行，避免与 A 同时测显存 |
| D | Docker、提交打包、文档 | 无 | 始终并行 |
| E | 新骨干短程 smoke/10 epoch | 高 | 只有空闲 GPU 或 A 完成后运行 |

若只有一张 5090，最有效的顺序是：A 独占训练，D 同时进行，B 在 A 的验证/结束后运行，C 和 E 排队。若确有多张 GPU，再将 B、C、E 分配到不同 GPU，并为每个进程固定 `CUDA_VISIBLE_DEVICES`。

### 4.2 可以并行的实验组合

以下组合互不改变同一个实验的归因，可以同时准备或运行：

- exp004 scheduler 训练 + Docker 文件和复现 README；
- exp004 scheduler 训练 + exp003 checkpoint 的 TTA/stressed 诊断；
- exp004 结果等待期间 + batch benchmark；
- exp005 训练准备 + 官方权重许可证核查 + 技术方案材料整理。

以下组合不建议同时运行或合并判断：

- scheduler、batch、增强、Dice、骨干同时修改；
- 两个训练进程同时占用同一张 5090；
- 只因 test_1 一次上涨就把所有改动合并为最终配方；
- 用 test_1 分数直接选择某个类别权重或阈值。

### 4.3 并行后的统一记录

每个作业必须有独立实验编号、输出目录、日志和配置。额外记录：

```text
job_id、CUDA_VISIBLE_DEVICES、启动时间、结束时间、GPU 型号、峰值显存、父实验、唯一变量
```

这样即使多个作业并行，也不会把日志、checkpoint 或排行榜结果混到同一个实验里。

### 4.4 可直接执行的诊断命令

exp004 训练独占 GPU 时，以下命令可以在另一张 GPU 或训练结束后执行：

```bash
export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=1 python src/compare_inference.py \
	--data-root /root/autodl-tmp/2026-低空图像语义分割赛道-训练集/train/train \
	--val-csv /root/autodl-tmp/val.csv \
	--checkpoint /root/autodl-tmp/runs/exp003_duration/best_model \
	--image-size 768 \
	--stride 384 \
	--device cuda \
	--output /root/autodl-tmp/analysis/exp003_inference_tta.json

CUDA_VISIBLE_DEVICES=1 python src/stressed_validation.py \
	--data-root /root/autodl-tmp/2026-低空图像语义分割赛道-训练集/train/train \
	--val-csv /root/autodl-tmp/val.csv \
	--checkpoint /root/autodl-tmp/runs/exp003_duration/best_model \
	--image-size 768 \
	--color 0.85 \
	--gamma 1.15 \
	--blur 0.6 \
	--noise 0.01 \
	--device cuda \
	--output /root/autodl-tmp/analysis/exp003_stressed_extended.json

CUDA_VISIBLE_DEVICES=1 python src/benchmark_batch.py \
	--checkpoint /root/autodl-tmp/runs/exp003_duration/best_model \
	--image-size 768 \
	--batch-sizes 2 4 8 16 \
	--device cuda
```

如果只有一张 5090，不要同时执行这些命令和训练；按 A 队列训练、B 队列诊断的顺序排队即可。若有第二张 GPU，将 `CUDA_VISIBLE_DEVICES=1` 改为实际空闲卡号。

## 5. exp004 scheduler：已完成

### 3.1 配置

exp004 只改变学习率策略，其他参数完全复用 exp003：

| 项目 | exp004 |
|---|---|
| 模型 | SegFormer MIT-B0 |
| 输入尺寸 | 768 |
| batch size | 2 |
| 初始学习率 | 6e-5 |
| 优化器 | AdamW，weight decay 0.01 |
| loss | CrossEntropy，Ignore=0 |
| 增强 | 水平翻转 |
| epoch | 60 |
| 新变量 | warmup 2 epoch + cosine decay |

建议先实现并检查 scheduler 的每 epoch 学习率记录，再启动训练。不要同时改 batch、输入尺寸、增强或 loss。

### 3.2 判定标准

- `val mIoU >= 0.782`：保留为新基准，继续测试后续变量；
- `0.779 <= val mIoU < 0.782`：视为基本持平，检查每类 IoU 和曲线后再决定是否保留；
- `val mIoU < 0.779`：不作为主配方，保留日志用于诊断；
- 若 Barren、Vehicle 或 Background 明显恶化，即使总 mIoU 小幅上升，也不能直接合并。

exp004 已完成，最佳验证集 mIoU 为 `0.7859696952`（epoch 56），相对 exp003 提升 `+0.0068576107`，8 个有效类别全部提升，因此保留为新的本地验证基准。完整事实记录见 [docs/analysis/exp004/exp004_scheduler.md](../analysis/exp004/exp004_scheduler.md)。

exp004 的 AutoDL 训练产物位于 `/root/autodl-tmp/runs/exp004_scheduler/`，本地轻量归档位于 `outputs/results/exp004_scheduler/`。下一步先用 exp004 epoch 56 的 `best_model` 完成推理策略对比，再按固定 768 resize 或验证有效的 TTA 提交一次 test_1，校准 scheduler 的线上收益，然后进入多尺度和受控域增强实验。

### 5.3 历史启动命令

```bash
export HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 python src/train.py \
	--data-root /root/autodl-tmp/2026-低空图像语义分割赛道-训练集/train/train \
	--train-csv /root/autodl-tmp/train.csv \
	--val-csv /root/autodl-tmp/val.csv \
	--pretrained /root/autodl-tmp/models/mit-b0 \
	--output-dir /root/autodl-tmp/runs/exp004_scheduler \
	--image-size 768 \
	--batch-size 2 \
	--epochs 60 \
	--learning-rate 6e-5 \
	--scheduler cosine \
	--warmup-epochs 2 \
	--device cuda \
	--log-file /root/autodl-tmp/runs/exp004_scheduler/training.log
```

`history.jsonl` 的每一轮记录会包含 `learning_rate`。启动前确认 `exp004_scheduler` 不含旧实验记录；训练结束后同时保存 `summary.json`、`best_metrics.json`、`best_model/` 和两个 checkpoint。

## 6. 并行的零训练工作

### 6.1 exp004 翻转 TTA

使用 exp004 checkpoint 在固定验证集比较：

- 原始 768 resize；
- 原始 + 水平翻转 TTA；
- 原始 + 水平/垂直翻转 TTA。

只有在固定验证集上有稳定增益时才并入复赛推理 SOP。建议门槛：

- 增益 `>= 0.002` 且关键类别没有明显下降：可采用；
- 增益小于 `0.002`：默认不增加推理复杂度；
- 结果下降：继续使用原始 768 resize。

此前 inference-001 已证明当前模型的 1024 全图和 768 滑窗不是免费收益：768 resize 明显更好。因此 exp003 阶段不把滑窗直接用于正式提交。等多尺度模型训练完成后，再重新测滑窗。

### 6.2 扩展 stressed validation

在 exp004 checkpoint 上运行以下受控场景：

- 色彩强度变化；
- gamma 变化；
- 轻度模糊；
- 轻度噪声或压缩扰动；
- 必要时再加入 RGB 通道增益和色温偏移。

用途是找出相对敏感的域变化，而不是预测排行榜分数。结果只用于选择增强项，并且要同时保留原始验证集结果，避免为了 stressed 分数牺牲正常场景能力。

### 6.3 batch 吞吐 benchmark

在 RTX 5090 上测试 batch size `2/4/8/16`，记录：

- 是否 OOM；
- 峰值显存；
- 每 step 耗时；
- images/s；
- 增大 batch 后是否需要线性调整学习率。

benchmark 使用随机输入，只用于硬件规划，不计入实验指标。建议在 exp004 结果确定后再决定是否做吞吐配方实验，避免把硬件优化和 scheduler 的效果混在一起。

## 7. 第二阶段训练顺序

### exp004b：吞吐配方

只有在 benchmark 证明 batch 增大可靠，并且 exp004 已经确定 scheduler 方向后执行。候选配置为 `batch size=8`，学习率按实际有效 batch 规模谨慎放大，仍使用 cosine。

这是一个配方级实验，不再声称只有一个变量。判定重点：

- 总 mIoU 相对 exp004 掉幅不超过 `0.003`；
- 训练耗时显著下降；
- 无类别短板明显恶化。

如果 batch 8 只带来小幅吞吐收益或显著改变收敛行为，就继续使用 batch 2，不为省时强行切换。

### exp005：多尺度与域增强

这是当前最重要的训练实验，也是唯一直接攻击训练域与官方测试域差距的主线。GLM 与 Copilot 对此方向一致，但不建议把所有增强一次性堆入；应先采用一个可解释、可复现的增强组：

- 随机尺度范围约 `0.5–1.25`；
- 随机裁剪或保持 768 输出尺寸；
- 依据 stressed validation 选择 hue、饱和度、gamma 或通道增益中的少量项；
- 保留水平翻转；
- 垂直翻转只在俯拍语义合理且验证结果支持时加入。

不要一次加入全部颜色扰动、强模糊、强噪声和复杂几何变换。增强必须防止把真实航拍纹理破坏成训练分布之外的图像。

判定标准：

- 原始 val mIoU 不低于当前最佳超过 `0.003` 的回退容忍度；
- stressed val 至少在目标扰动上改善；
- Barren、Vehicle、Road 至少有一个明确改善，且其他类别没有大幅下降；
- 重新比较 resize、滑窗和 TTA，不能沿用旧模型的推理结论。

### exp006：CE + 温和 Dice

如果 exp005 仍显示 Barren 被 Vegetation 吞并，再测试 CE + 温和 Dice。Dice 权重从小值开始，避免少数类提升时破坏 Building、Road 和 Vegetation。

门槛：

- Barren 明显提升，目标是接近或超过 `0.58`；
- 总 mIoU 不低于 exp005；
- Vehicle 和 Road 不出现明显回退。

不建议在此阶段同时加入激进类别权重和过采样。当前 test_1 的 Barren 预测占比偏低，但这不足以单独证明激进加权一定有效。

### exp007：更大骨干

在 scheduler、增强和 loss 方向稳定后，再比较 MIT-B1 或 MIT-B2。更大骨干应单独作为一次对照，至少保留 MIT-B0 作为可靠 fallback。

GLM 建议的遥感预训练骨干可以作为探索方向，但优先级低于 exp005 的多尺度域增强，先做 mit-b2 同构快验。任何遥感预训练骨干必须先满足三个条件：

1. 权重来源公开、学术许可清楚；
2. 不使用额外比赛数据或未授权数据；
3. 结构和训练命令能被 Docker 从零复现。

由于新骨干有较高工程风险，先做短程可行性验证，不要直接把它和多尺度、强增强、Dice 一起投入完整训练。

## 8. 0.72 冲刺线的资源排序

当时间有限时，按以下顺序投入：

1. S2 提交 exp004，获得 scheduler 的线上校准；
2. stressed validation 和 TTA，确定零训练收益；
3. exp005 多尺度 + 受控域增强；
4. exp005 后重测滑窗/TTA，再决定推理方案；
5. exp006 CE + 温和 Dice；
6. exp007 MIT-B1/B2 快验；
7. 公开遥感预训练骨干短程验证；
8. EMA 和最终组合长训。

EMA 可以放在最终候选配方确认后加入，因为它通常是低风险的稳定化手段，但必须用固定验证集比较原始 best 与 EMA 权重，不能默认有效。

不建议把伪标签自训练纳入当前默认计划。测试图伪标签即使技术上可行，也存在规则解释和复现风险；只有取得组委会书面确认后才讨论实现。

## 9. test_1 提交策略与复赛准备

### 9.1 test_1 的正确用途

既然提交次数没有限制，可以把 test_1 作为“外部验证集”，但必须设置提交闸门：

1. 先在固定 val 和 stressed val 上确认候选配方不是偶然回退；
2. 只有完成一个明确实验或推理变体后才提交一次；
3. 提交前登记候选名称、唯一变化、预期方向和验证集结果；
4. 提交后只记录分数，不再针对该次分数局部调阈值或调类别权重；
5. 至少间隔一个完整实验再使用 test_1 结果决定方向。

建议提交节奏：

| 提交 | 内容 | 目的 |
|---|---|---|
| S0 | exp002 | 已有线上基准 0.591572 |
| S1 | exp003 | 已有校准结果 0.608668 |
| S2 | exp004 最佳 checkpoint，768 resize | 验证 scheduler 是否延续收益 |
| S3 | exp005 最佳 checkpoint，验证后选择 TTA/resize | 验证多尺度和域增强 |
| S4 | 最终候选模型 | 服务复赛/半决赛方案，不再为 test_1 过拟合 |

TTA、滑窗或其他推理变体可以单独提交，但每次必须同时保留对应的本地验证结果和 ZIP 哈希。当前 exp003 的默认策略仍是 768 resize；滑窗只有在多尺度模型的固定验证集上重新证明有效后才升级。

### 9.2 不要把 test_1 无限提交误用成测试集调参

无限次数提高了诊断能力，但不改变科学性边界。若连续提交很多近似配方并只保留 test_1 最高分，会产生测试集过拟合，最终可能在 test_2 和 test_3 失效。test_1 应用于区分“大方向是否有效”，而不是搜索最后一个学习率小数位。

## 10. 复赛提交与 Docker 线

训练线推进的同时完成以下工程闭环：

- Docker 镜像可构建；
- 干净环境能运行 smoke test；
- 数据挂载后能完成预测；
- `check_submission.py` 能检查 500/1300 张测试图；
- `package_submission.py` 将 PNG 直接写入 ZIP 根目录；
- 记录 checkpoint、输入尺寸、推理策略、提交时间和 ZIP SHA-256；
- 用一个全新的空目录做一次从数据到 ZIP 的复现演练。

提交格式固定为：

```text
submission.zip
├── test1_1.png
├── test1_2.png
├── ...
└── test1_500.png
```

ZIP 内不能出现 `test_1/` 或其他子文件夹。预测 PNG 必须保持原测试图尺寸、单通道 `L` 模式和合法类别 ID。

## 11. 结果记录模板

每次实验结束后，至少记录：

```text
实验编号：
父实验：
唯一变量：
固定变量：
最佳 epoch：
最佳 val mIoU：
8 类 IoU：
预测 Ignore 比例：
stressed val：
训练耗时：
推理策略：
是否保留：
保留/淘汰原因：
```

线上提交只记录为校准证据，不把无标签 test 结果当作训练标签。exp003 已经完成了 test_1 校准，下一次正式提交应优先服务复赛，不再反复消耗 test_1 提交次数。

## 12. 最终决策

exp004 已经完成并应保留。当前立即执行 S2 推理校准和 exp005 训练准备；诊断、Docker、规则核查和文档线并行推进。暂缓：

- 继续使用恒定学习率堆到 80/100 epoch；
- 在没有实验依据时直接使用滑窗提交；
- 同时改变 scheduler、batch、增强、loss 和模型规模；
- 未经规则确认的伪标签自训练；
- 多模型集成或权重平均。

本阶段的成功标准不是一次实验直接达到 0.72，而是建立一条能在复赛和半决赛持续迭代、可复现、可解释、不会因提交格式或规则风险失分的训练与提交链路。若连续两轮可靠实验仍只能维持 `0.61–0.63`，应重新审视数据域差距和类别映射，而不是继续无方向增加训练时间。
