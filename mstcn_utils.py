"""Loss, metrics and checkpoint helpers shared by MS-TCN scripts."""
import os

import mxnet as mx
import numpy as np


def masked_focal_loss(logits, labels, mask, class_weights, gamma=0.0):
    """Sparse class-weighted focal loss for ``NCT`` logits and ``NT`` labels."""
    if gamma < 0:
        raise ValueError('gamma must be non-negative')
    class_count = int(logits.shape[1])
    log_probabilities = mx.nd.log_softmax(logits, axis=1)
    flat_log_probabilities = log_probabilities.transpose(
        (0, 2, 1)).reshape((-1, class_count))
    flat_labels = labels.astype('int32').reshape((-1,))
    flat_mask = mask.reshape((-1,))
    log_pt = mx.nd.pick(
        flat_log_probabilities, flat_labels, axis=1, keepdims=False)
    pt = mx.nd.exp(log_pt)
    sample_weights = mx.nd.take(class_weights, flat_labels)
    losses = -sample_weights * mx.nd.power(1.0 - pt, gamma) * log_pt
    denominator = mx.nd.maximum(flat_mask.sum(), mx.nd.ones((1,), ctx=mask.context))
    return (losses * flat_mask).sum() / denominator


def temporal_smoothing_loss(logits, mask, cap=4.0):
    """Truncated MSE between adjacent log-probabilities within valid regions."""
    if cap <= 0:
        raise ValueError('smoothing cap must be positive')
    if int(logits.shape[2]) < 2:
        return mx.nd.zeros((1,), ctx=logits.context).sum()

    log_probabilities = mx.nd.log_softmax(logits, axis=1)
    current = mx.nd.slice_axis(
        log_probabilities, axis=2, begin=1, end=None)
    previous = mx.nd.stop_gradient(mx.nd.slice_axis(
        log_probabilities, axis=2, begin=0, end=-1))
    squared_difference = mx.nd.square(current - previous)
    squared_difference = mx.nd.minimum(
        squared_difference,
        mx.nd.ones_like(squared_difference) * cap)

    adjacent_mask = mask[:, 1:] * mask[:, :-1]
    expanded_mask = adjacent_mask.expand_dims(axis=1)
    denominator = adjacent_mask.sum() * int(logits.shape[1])
    denominator = mx.nd.maximum(
        denominator, mx.nd.ones((1,), ctx=mask.context))
    return (squared_difference * expanded_mask).sum() / denominator


def multi_stage_loss(stage_logits, labels, mask, class_weights, gamma=0.0,
                     smoothing_weight=0.15, smoothing_cap=4.0):
    """Average classification and smoothing losses over all MS-TCN stages."""
    stage_count = int(stage_logits.shape[1])
    total_classification = None
    total_smoothing = None
    for stage_index in range(stage_count):
        logits = stage_logits[:, stage_index, :, :]
        classification = masked_focal_loss(
            logits, labels, mask, class_weights, gamma=gamma)
        smoothing = temporal_smoothing_loss(
            logits, mask, cap=smoothing_cap)
        total_classification = (
            classification if total_classification is None
            else total_classification + classification)
        total_smoothing = (
            smoothing if total_smoothing is None
            else total_smoothing + smoothing)

    classification_loss = total_classification / stage_count
    smoothing_loss = total_smoothing / stage_count
    total_loss = classification_loss + smoothing_weight * smoothing_loss
    return total_loss, classification_loss, smoothing_loss


def final_stage(stage_logits):
    """Return final-stage logits with shape ``(batch, classes, time)``."""
    return stage_logits[:, int(stage_logits.shape[1]) - 1, :, :]


def update_confusion(matrix, logits, labels, mask):
    """Accumulate a NumPy confusion matrix from masked sequence predictions."""
    predictions = logits.argmax(axis=1).asnumpy().astype(np.int32)
    labels_np = labels.asnumpy().astype(np.int32)
    valid = mask.asnumpy() > 0.5
    np.add.at(matrix, (labels_np[valid], predictions[valid]), 1)


def confusion_metrics(matrix, class_names, background_index=0):
    """Compute frame metrics, including foreground macro F1."""
    per_class = []
    for class_index, class_name in enumerate(class_names):
        true_positive = int(matrix[class_index, class_index])
        false_positive = int(matrix[:, class_index].sum() - true_positive)
        false_negative = int(matrix[class_index, :].sum() - true_positive)
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        per_class.append({
            'class_index': class_index,
            'class_name': str(class_name),
            'support': int(matrix[class_index, :].sum()),
            'predicted': int(matrix[:, class_index].sum()),
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

    foreground = [
        row for row in per_class
        if row['class_index'] != background_index
    ]
    total = int(matrix.sum())
    metrics = {
        'samples': total,
        'accuracy': _safe_divide(np.trace(matrix), total),
        'macro_f1': float(np.mean([row['f1'] for row in per_class])),
        'foreground_macro_f1': float(np.mean(
            [row['f1'] for row in foreground])) if foreground else 0.0,
    }
    return metrics, per_class


def find_latest_checkpoint(experiment_dir):
    checkpoints = _numeric_checkpoints(experiment_dir)
    return checkpoints[-1][1] if checkpoints else None


def find_best_checkpoint(experiment_dir):
    scores_path = os.path.join(experiment_dir, 'scores.txt')
    scored_epochs = []
    if os.path.exists(scores_path):
        with open(scores_path, 'r') as score_file:
            for line in score_file:
                fields = line.strip().split()
                if len(fields) < 2:
                    continue
                try:
                    score = float(fields[1])
                    if np.isfinite(score):
                        scored_epochs.append((score, int(fields[0])))
                except ValueError:
                    continue
    if scored_epochs:
        _, best_epoch = max(scored_epochs)
        path = os.path.join(experiment_dir, '{:04d}.params'.format(best_epoch))
        if os.path.exists(path):
            return path
    return find_latest_checkpoint(experiment_dir)


def checkpoint_epoch(path):
    if path is None:
        return -1
    stem = os.path.splitext(os.path.basename(path))[0]
    return int(stem) if stem.isdigit() else -1


def _numeric_checkpoints(experiment_dir):
    if not os.path.isdir(experiment_dir):
        return []
    checkpoints = []
    for filename in os.listdir(experiment_dir):
        stem, extension = os.path.splitext(filename)
        if extension == '.params' and stem.isdigit():
            checkpoints.append((int(stem), os.path.join(experiment_dir, filename)))
    return sorted(checkpoints)


def _safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0
