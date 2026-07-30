# 本地改进功能

本文档记录在原项目基础上新增、且与旧命令兼容的训练和评估选项。

## 标签感知的水平翻转

训练默认启用 `--horizontal_flip`。图像水平翻转时会同步交换以下类别：

```text
HFL <-> HFR
HNL <-> HNR
```

原项目只翻转图像、不交换标签，会给左右击球类别引入错误监督。需要关闭翻转进行消融实验时使用：

```bash
python train.py \
  --model_id 0008 \
  --backbone DenseNet121 \
  --nohorizontal_flip
```

## 可复现随机种子

训练和评估新增 `--seed`，默认值为 `42`：

```bash
python train.py --model_id 0008 --backbone DenseNet121 --seed 42
```

它会同时设置 Python、NumPy 和 MXNet 的随机种子。

## 类别加权与 Focal Loss

默认仍然使用原项目的普通交叉熵。可选的类别权重为：

- `none`：不加权，默认值。
- `inverse_sqrt`：类别样本数平方根的倒数。
- `effective_num`：按有效样本数计算权重。

例如：

```bash
python train.py \
  --model_id 0009 \
  --backbone DenseNet121 \
  --class_weighting inverse_sqrt \
  --focal_gamma 2.0 \
  --seed 42
```

建议先分别比较“仅加权”和“仅 Focal Loss”，再测试二者组合：

```bash
# 仅类别加权
python train.py --model_id 0009 --backbone DenseNet121 \
  --class_weighting inverse_sqrt --focal_gamma 0

# 仅 Focal Loss
python train.py --model_id 0010 --backbone DenseNet121 \
  --class_weighting none --focal_gamma 2.0
```

类别权重使用实际训练样本统计计算，并归一化为均值 1；具体权重会写入实验日志。

## 导出逐帧预测

评估时增加 `--save_predictions`：

```bash
python evaluate.py \
  --model_id 0007 \
  --backbone DenseNet121 \
  --save_predictions
```

默认输出：

```text
models/vision/experiments/0007/predictions_test.npz
```

也可以指定路径：

```bash
python evaluate.py \
  --model_id 0007 \
  --backbone DenseNet121 \
  --predictions_file analysis/0007_test.npz
```

NPZ 文件包含：

- `paths`：对应帧文件路径
- `videos`：视频 ID
- `frames`：帧编号
- `logits`：模型原始输出
- `probabilities`：11 类 softmax 概率
- `labels`：真实类别编号
- `predictions`：预测类别编号
- `class_names`：类别名称与编号的对应关系

这些数据可以直接用于混淆矩阵、逐视频分析、时序平滑和事件级指标计算。
