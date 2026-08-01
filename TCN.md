# TCN 时序模型

TCN 使用 006/007 等帧级 CNN 导出的特征，在时间窗口上执行残差空洞一维卷积，
最后预测窗口中心帧的 11 类事件。它不需要重复运行 DenseNet 主干，适合快速验证时序上下文。

## 1. 导出最佳帧模型特征

以 0007 为例：

```bash
python train.py \
  --model_id 0007 \
  --backbone DenseNet121 \
  --save_feats
```

这里使用 `train.py`，因为它会为 train、val、test 三个划分都导出特征；
`evaluate.py --save_feats` 只导出当前评估划分，不足以训练时序模型。

确保特征位于：

```text
data/features/0007/
```

## 2. 训练 TCN

```bash
python train.py \
  --model_id 0011 \
  --backbone DenseNet121 \
  --feats_model 0007 \
  --temp_pool tcn \
  --window 61 \
  --stride 1 \
  --tcn_hidden 128 \
  --tcn_layers 4 \
  --tcn_kernel_size 3 \
  --tcn_dropout 0.2 \
  --class_weighting inverse_sqrt \
  --focal_gamma 0 \
  --seed 42
```

默认会从首个测试样本自动检测保存的特征形状。只有需要跳过自动检测时，才传入
例如 `--feature_size 4096` 的显式展平维度。

4 层、卷积核 3 的默认 TCN 使用膨胀率 1、2、4、8；每层包含两个卷积，
因此感受野覆盖 61 帧。
窗口可以大于感受野，但建议先比较 `31`、`61` 和 `121`。

## 3. 评估与导出预测

评估参数必须与训练时一致：

```bash
python evaluate.py \
  --model_id 0011 \
  --backbone DenseNet121 \
  --feats_model 0007 \
  --temp_pool tcn \
  --window 61 \
  --tcn_hidden 128 \
  --tcn_layers 4 \
  --tcn_kernel_size 3 \
  --tcn_dropout 0.2 \
  --save_predictions
```

`evaluate.py` 现在会正确使用 `--num_gpus`；传 `--num_gpus 0` 可强制使用 CPU。

然后使用 `analyze_predictions.py` 比较帧级和事件级结果：

```bash
python analyze_predictions.py \
  models/vision/experiments/0011/predictions_test.npz
```

## 建议的消融实验

保持随机种子和其他参数不变，仅修改一个变量：

- `window`: 31 / 61 / 121
- `tcn_layers`: 3 / 4 / 5
- `class_weighting`: none / inverse_sqrt
- `focal_gamma`: 0 / 2

主要比较 `foreground_macro_f1`、`event F1 @ tIoU 0.5` 和 `mAP @ tIoU 0.5`。
