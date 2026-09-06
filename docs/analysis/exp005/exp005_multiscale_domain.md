# exp005 多尺度与受控域增强

> 实验定义日期：2026-09-06。当前仅完成实现验证和第 1 个 epoch，完整训练已暂停，等待 exp004 S2 排行榜结果和队友评审。

## 实验目的

在 exp004 scheduler 配方上只改变训练数据增强，验证多尺度裁剪和温和颜色扰动能否改善尺度变化与 test_1 域偏移。验证集仍使用确定性的 768 resize，保持与 exp004 指标口径一致。

## 配置

| 项目 | exp004 | exp005 |
|---|---|---|
| 模型 | SegFormer MIT-B0 | SegFormer MIT-B0 |
| 输入尺寸 | 768 x 768 | 768 x 768 |
| batch size | 2 | 2 |
| epoch | 60 | 60 |
| 学习率 | 6e-5 | 6e-5 |
| scheduler | warmup 2 + cosine | warmup 2 + cosine |
| 基础翻转 | 水平翻转 | 水平 + 垂直翻转 |
| 多尺度 | 无 | 原图随机缩放 0.5-1.25，随机裁剪/填充到 768 |
| 颜色增强 | 无 | brightness/contrast/saturation，各自 0.9-1.1 |
| loss | CrossEntropy，Ignore=0 | CrossEntropy，Ignore=0 |
| 数据划分 | 固定 5597 / 1399 | 固定 5597 / 1399 |

本实验不加入 Dice、类别权重、EMA 或更大骨干，避免失去归因。

## 实现验证

- 6 项单元测试通过：增强输出尺寸、标签范围、验证确定性、mIoU Ignore 逻辑和 checkpoint 恢复兼容性。
- AutoDL 小型 split 真实训练烟雾测试通过：完成 SegFormer 前向、CE、反向、验证和 checkpoint 保存。
- 旧实验默认行为保持不变；只有显式传入 `--multiscale`、`--color-jitter` 和 `--vertical-flip` 才启用新增增强。

## AutoDL 命令

```bash
python /root/autodl-tmp/src/train.py \
  --data-root /root/autodl-tmp/2026-低空图像语义分割赛道-训练集/train/train \
  --train-csv /root/autodl-tmp/train.csv \
  --val-csv /root/autodl-tmp/val.csv \
  --output-dir /root/autodl-tmp/runs/exp005_multiscale_domain \
  --image-size 768 \
  --batch-size 2 \
  --epochs 60 \
  --learning-rate 6e-5 \
  --scheduler cosine \
  --warmup-epochs 2 \
  --multiscale \
  --scale-min 0.5 \
  --scale-max 1.25 \
  --color-jitter 0.1 \
  --vertical-flip \
  --num-workers 2 \
  --device cuda \
  --pretrained /root/autodl-tmp/models/mit-b0 \
  --log-file /root/autodl-tmp/runs/exp005_multiscale_domain/training.log
```

## 判定标准

1. 原始 val mIoU 相对 exp004 `0.7859696952` 的回退不超过 `0.003`。
2. 优先观察 Vehicle、Road、Barren 是否至少一项明确改善。
3. 训练完成后重新比较 resize、水平 TTA 和滑窗，不沿用 exp004 的推理结论。
4. 通过本地和 stressed validation 后再生成 S3，不根据 test_1 单次分数调整类别阈值。

## 结果

全量训练曾启动并完成 epoch 1：

- train loss：`1.3253143426`
- val mIoU：`0.4367515908`
- epoch 耗时：`221.83` 秒

该早期结果只证明全量数据训练链路正常，不用于判断增强有效性。实验已在 epoch 1 后主动停止，待以下信息齐备后再确认是否恢复：

1. exp004 S2 的 test_1 排行榜成绩；
2. 队友对尺度下限 `0.5`、垂直翻转和颜色强度 `0.1` 的评审意见；
3. 是否先运行 exp004 stressed validation 来缩小颜色增强范围。