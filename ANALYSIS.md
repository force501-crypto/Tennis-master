# 预测结果分析与时序后处理

先通过评估脚本导出逐帧预测：

```bash
python evaluate.py \
  --model_id 0007 \
  --backbone DenseNet121 \
  --save_predictions
```

然后生成原始分析报告：

```bash
python analyze_predictions.py \
  models/vision/experiments/0007/predictions_test.npz
```

尝试概率平滑、合并短间隔并过滤过短事件。`--smooth-window` 必须使用正奇数：

```bash
python analyze_predictions.py \
  models/vision/experiments/0007/predictions_test.npz \
  --smooth-window 9 \
  --merge-gap 3 \
  --min-event-length 5
```

默认会在预测文件旁生成 `analysis_predictions_test/`，其中包含：

- `summary.json`：原始与后处理后的帧级汇总，以及事件级指标。
- `per_class_raw.csv` / `per_class_processed.csv`：每类 Precision、Recall、F1、样本数。
- `per_video.csv`：每场比赛在处理前后的帧级指标。
- `confusion_matrix_raw.csv` / `confusion_matrix_processed.csv`：真实类别为行、预测类别为列。
- `event_metrics_raw.csv` / `event_metrics_processed.csv`：各类别在 tIoU 0.3、0.5、0.7 下的事件指标和 AP。
- `events_raw.csv` / `events_processed.csv`：真实与预测事件的起止帧。
- `processed_predictions.npz`：平滑和过滤后的概率及预测。

建议先保存完全不做后处理的报告作为基线，再单独调整一个参数。重点比较：

- `foreground_macro_f1`
- `event F1 @ tIoU 0.5`
- `mAP @ tIoU 0.5`

后处理参数必须只用验证集选择，最终确定后再应用到测试集，避免利用测试结果调参。

## 历史指标说明

原项目 `metrics/vision.py` 中 Precision 和 Recall 的分母写反，因此旧的 006/007
日志里每类 Precision 与 Recall 的名称需要对调理解；F1 的计算结果不受这个错误影响。
当前代码已经修正，新生成的日志和本分析脚本都使用标准定义：

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```
