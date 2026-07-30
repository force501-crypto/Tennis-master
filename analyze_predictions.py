"""Analyse exported frame predictions and evaluate temporal events."""
import argparse
import csv
import json
import os

import numpy as np


REQUIRED_KEYS = {
    'videos', 'frames', 'probabilities', 'labels', 'predictions', 'class_names'
}


def safe_divide(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        missing = REQUIRED_KEYS.difference(data.files)
        if missing:
            raise ValueError('Prediction file is missing keys: {}'.format(sorted(missing)))
        loaded = {key: np.asarray(data[key]) for key in data.files}

    loaded['videos'] = loaded['videos'].astype(str)
    loaded['class_names'] = loaded['class_names'].astype(str)
    loaded['frames'] = loaded['frames'].astype(np.int64)
    loaded['labels'] = loaded['labels'].astype(np.int32)
    loaded['predictions'] = loaded['predictions'].astype(np.int32)
    loaded['probabilities'] = loaded['probabilities'].astype(np.float32)

    sample_count = len(loaded['labels'])
    for key in ('videos', 'frames', 'predictions', 'probabilities'):
        if len(loaded[key]) != sample_count:
            raise ValueError('{} has a different sample count'.format(key))
    if loaded['probabilities'].shape != (sample_count, len(loaded['class_names'])):
        raise ValueError('probabilities shape does not match samples and classes')
    if np.any(loaded['labels'] < 0) or np.any(loaded['labels'] >= len(loaded['class_names'])):
        raise ValueError('labels contain an invalid class index')

    order = np.lexsort((loaded['frames'], loaded['videos']))
    for key in ('videos', 'frames', 'labels', 'predictions', 'probabilities'):
        loaded[key] = loaded[key][order]
    if 'paths' in loaded:
        loaded['paths'] = loaded['paths'].astype(str)[order]
    if 'logits' in loaded:
        loaded['logits'] = loaded['logits'][order]
    return loaded


def infer_frame_step(videos, frames):
    differences = []
    for video in np.unique(videos):
        video_frames = frames[videos == video]
        diff = np.diff(video_frames)
        differences.extend(diff[diff > 0].tolist())
    if not differences:
        return 1
    values, counts = np.unique(np.asarray(differences, dtype=np.int64), return_counts=True)
    max_count = counts.max()
    return int(values[counts == max_count].min())


def contiguous_runs(videos, frames, frame_step):
    if len(frames) == 0:
        return []
    breaks = np.where(
        (videos[1:] != videos[:-1]) |
        ((frames[1:] - frames[:-1]) > frame_step) |
        ((frames[1:] - frames[:-1]) <= 0)
    )[0] + 1
    boundaries = np.concatenate(([0], breaks, [len(frames)]))
    return [(int(boundaries[i]), int(boundaries[i + 1]))
            for i in range(len(boundaries) - 1)]


def smooth_probabilities(probabilities, runs, window):
    if window < 1 or window % 2 == 0:
        raise ValueError('smooth_window must be a positive odd integer')
    smoothed = probabilities.copy()
    if window == 1:
        return smoothed

    radius = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    for start, end in runs:
        segment = probabilities[start:end]
        if len(segment) == 0:
            continue
        padded = np.pad(segment, ((radius, radius), (0, 0)), mode='edge')
        for class_index in range(segment.shape[1]):
            smoothed[start:end, class_index] = np.convolve(
                padded[:, class_index], kernel, mode='valid')
    smoothed /= smoothed.sum(axis=1, keepdims=True)
    return smoothed


def label_segments(labels, start, end):
    if end <= start:
        return []
    changes = np.where(labels[start + 1:end] != labels[start:end - 1])[0] + start + 1
    boundaries = np.concatenate(([start], changes, [end]))
    return [(int(boundaries[i]), int(boundaries[i + 1]), int(labels[boundaries[i]]))
            for i in range(len(boundaries) - 1)]


def merge_short_background_gaps(labels, runs, background_index, max_gap_frames, frame_step):
    merged = labels.copy()
    if max_gap_frames <= 0:
        return merged
    for run_start, run_end in runs:
        segments = label_segments(merged, run_start, run_end)
        for segment_index in range(1, len(segments) - 1):
            start, end, class_index = segments[segment_index]
            previous_class = segments[segment_index - 1][2]
            next_class = segments[segment_index + 1][2]
            duration = (end - start) * frame_step
            if (class_index == background_index and duration <= max_gap_frames
                    and previous_class == next_class and previous_class != background_index):
                merged[start:end] = previous_class
    return merged


def remove_short_events(labels, runs, background_index, min_event_frames, frame_step):
    filtered = labels.copy()
    if min_event_frames <= 1:
        return filtered
    for run_start, run_end in runs:
        for start, end, class_index in label_segments(filtered, run_start, run_end):
            duration = (end - start) * frame_step
            if class_index != background_index and duration < min_event_frames:
                filtered[start:end] = background_index
    return filtered


def confusion_matrix(labels, predictions, class_count):
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def frame_metrics(labels, predictions, class_names, background_index):
    matrix = confusion_matrix(labels, predictions, len(class_names))
    per_class = []
    for class_index, class_name in enumerate(class_names):
        true_positive = int(matrix[class_index, class_index])
        false_positive = int(matrix[:, class_index].sum() - true_positive)
        false_negative = int(matrix[class_index, :].sum() - true_positive)
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2.0 * precision * recall, precision + recall)
        per_class.append({
            'class_index': class_index,
            'class_name': str(class_name),
            'support': int(matrix[class_index, :].sum()),
            'predicted': int(matrix[:, class_index].sum()),
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })

    foreground = [row for row in per_class if row['class_index'] != background_index]
    summary = {
        'samples': int(len(labels)),
        'accuracy': safe_divide(np.trace(matrix), matrix.sum()),
        'macro_precision': float(np.mean([row['precision'] for row in per_class])),
        'macro_recall': float(np.mean([row['recall'] for row in per_class])),
        'macro_f1': float(np.mean([row['f1'] for row in per_class])),
        'foreground_macro_precision': float(np.mean([row['precision'] for row in foreground])),
        'foreground_macro_recall': float(np.mean([row['recall'] for row in foreground])),
        'foreground_macro_f1': float(np.mean([row['f1'] for row in foreground])),
    }
    return summary, per_class, matrix


def per_video_metrics(videos, labels, predictions, class_names, background_index):
    rows = []
    for video in np.unique(videos):
        selection = videos == video
        summary, _, _ = frame_metrics(
            labels[selection], predictions[selection], class_names, background_index)
        summary['video'] = str(video)
        rows.append(summary)
    return rows


def combine_video_metrics(raw_rows, processed_rows):
    processed_by_video = {row['video']: row for row in processed_rows}
    combined = []
    for raw_row in raw_rows:
        video = raw_row['video']
        processed_row = processed_by_video[video]
        row = {'video': video, 'samples': raw_row['samples']}
        for key, value in raw_row.items():
            if key not in ('video', 'samples'):
                row['raw_{}'.format(key)] = value
        for key, value in processed_row.items():
            if key not in ('video', 'samples'):
                row['processed_{}'.format(key)] = value
        combined.append(row)
    return combined


def extract_events(videos, frames, labels, probabilities, runs, frame_step, background_index):
    events = []
    for run_start, run_end in runs:
        for start, end, class_index in label_segments(labels, run_start, run_end):
            if class_index == background_index:
                continue
            events.append({
                'video': str(videos[start]),
                'class_index': class_index,
                'start_frame': int(frames[start]),
                'end_frame': int(frames[end - 1] + frame_step),
                'score': float(probabilities[start:end, class_index].mean()),
            })
    return events


def temporal_iou(first, second):
    intersection = max(
        0, min(first['end_frame'], second['end_frame'])
        - max(first['start_frame'], second['start_frame']))
    union = (first['end_frame'] - first['start_frame']
             + second['end_frame'] - second['start_frame'] - intersection)
    return safe_divide(intersection, union)


def average_precision(recalls, precisions):
    if len(recalls) == 0:
        return 0.0
    recall_envelope = np.concatenate(([0.0], recalls, [1.0]))
    precision_envelope = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(len(precision_envelope) - 2, -1, -1):
        precision_envelope[index] = max(
            precision_envelope[index], precision_envelope[index + 1])
    changes = np.where(recall_envelope[1:] != recall_envelope[:-1])[0] + 1
    return float(np.sum(
        (recall_envelope[changes] - recall_envelope[changes - 1])
        * precision_envelope[changes]))


def match_events(predicted_events, ground_truth_events, threshold):
    predictions = sorted(predicted_events, key=lambda event: event['score'], reverse=True)
    matched_ground_truth = set()
    true_positives = np.zeros(len(predictions), dtype=np.float64)
    false_positives = np.zeros(len(predictions), dtype=np.float64)

    for prediction_index, prediction in enumerate(predictions):
        candidates = []
        for ground_truth_index, ground_truth in enumerate(ground_truth_events):
            if ground_truth_index in matched_ground_truth:
                continue
            if prediction['video'] != ground_truth['video']:
                continue
            candidates.append((temporal_iou(prediction, ground_truth), ground_truth_index))
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_iou >= threshold:
            true_positives[prediction_index] = 1.0
            matched_ground_truth.add(best_index)
        else:
            false_positives[prediction_index] = 1.0

    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recalls = cumulative_tp / max(len(ground_truth_events), 1)
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0)
    true_positive_count = int(true_positives.sum())
    precision = safe_divide(true_positive_count, len(predictions))
    recall = safe_divide(true_positive_count, len(ground_truth_events))
    return {
        'ground_truth_events': len(ground_truth_events),
        'predicted_events': len(predictions),
        'true_positives': true_positive_count,
        'precision': precision,
        'recall': recall,
        'f1': safe_divide(2.0 * precision * recall, precision + recall),
        'ap': average_precision(recalls, precisions) if ground_truth_events else None,
    }


def event_metrics(predicted_events, ground_truth_events, class_names,
                  background_index, thresholds):
    rows = []
    summaries = []
    foreground_classes = [
        index for index in range(len(class_names)) if index != background_index
    ]
    for threshold in thresholds:
        class_results = []
        for class_index in foreground_classes:
            predicted = [
                event for event in predicted_events if event['class_index'] == class_index
            ]
            ground_truth = [
                event for event in ground_truth_events if event['class_index'] == class_index
            ]
            result = match_events(predicted, ground_truth, threshold)
            result.update({
                'threshold': float(threshold),
                'class_index': class_index,
                'class_name': str(class_names[class_index]),
            })
            rows.append(result)
            class_results.append(result)

        total_gt = sum(result['ground_truth_events'] for result in class_results)
        total_pred = sum(result['predicted_events'] for result in class_results)
        total_tp = sum(result['true_positives'] for result in class_results)
        micro_precision = safe_divide(total_tp, total_pred)
        micro_recall = safe_divide(total_tp, total_gt)
        valid_ap = [result['ap'] for result in class_results if result['ap'] is not None]
        summaries.append({
            'threshold': float(threshold),
            'ground_truth_events': total_gt,
            'predicted_events': total_pred,
            'true_positives': total_tp,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': safe_divide(
                2.0 * micro_precision * micro_recall, micro_precision + micro_recall),
            'map': float(np.mean(valid_ap)) if valid_ap else 0.0,
        })
    return summaries, rows


def write_csv(path, rows, fieldnames=None):
    if fieldnames is None:
        if not rows:
            raise ValueError('fieldnames are required when rows is empty')
        fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix(path, matrix, class_names):
    with open(path, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['ground_truth/prediction'] + class_names.tolist())
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + row.tolist())


def write_event_csv(path, ground_truth_events, predicted_events, class_names):
    rows = []
    for source, events in (('ground_truth', ground_truth_events),
                           ('prediction', predicted_events)):
        for event in events:
            rows.append({
                'source': source,
                'video': event['video'],
                'class_index': event['class_index'],
                'class_name': str(class_names[event['class_index']]),
                'start_frame': event['start_frame'],
                'end_frame': event['end_frame'],
                'score': event['score'],
            })
    write_csv(
        path, rows,
        ['source', 'video', 'class_index', 'class_name',
         'start_frame', 'end_frame', 'score'])


def analyse(args):
    data = load_predictions(args.predictions_file)
    class_names = data['class_names']
    if args.background_class not in class_names:
        raise ValueError('Background class {} was not found'.format(args.background_class))
    background_index = int(np.where(class_names == args.background_class)[0][0])
    if args.frame_step is not None and args.frame_step <= 0:
        raise ValueError('frame_step must be positive')
    if args.merge_gap < 0:
        raise ValueError('merge_gap must be non-negative')
    if args.min_event_length < 1:
        raise ValueError('min_event_length must be at least 1')
    if any(threshold <= 0.0 or threshold > 1.0 for threshold in args.tiou_thresholds):
        raise ValueError('tIoU thresholds must be in (0, 1]')

    frame_step = args.frame_step or infer_frame_step(data['videos'], data['frames'])
    runs = contiguous_runs(data['videos'], data['frames'], frame_step)

    smoothed_probabilities = smooth_probabilities(
        data['probabilities'], runs, args.smooth_window)
    processed_predictions = smoothed_probabilities.argmax(axis=1).astype(np.int32)
    processed_predictions = merge_short_background_gaps(
        processed_predictions, runs, background_index, args.merge_gap, frame_step)
    processed_predictions = remove_short_events(
        processed_predictions, runs, background_index, args.min_event_length, frame_step)

    raw_summary, raw_per_class, raw_matrix = frame_metrics(
        data['labels'], data['predictions'], class_names, background_index)
    processed_summary, per_class, matrix = frame_metrics(
        data['labels'], processed_predictions, class_names, background_index)
    raw_video_rows = per_video_metrics(
        data['videos'], data['labels'], data['predictions'],
        class_names, background_index)
    processed_video_rows = per_video_metrics(
        data['videos'], data['labels'], processed_predictions,
        class_names, background_index)
    video_rows = combine_video_metrics(raw_video_rows, processed_video_rows)

    ground_truth_events = extract_events(
        data['videos'], data['frames'], data['labels'], data['probabilities'],
        runs, frame_step, background_index)
    predicted_events = extract_events(
        data['videos'], data['frames'], processed_predictions,
        smoothed_probabilities, runs, frame_step, background_index)
    raw_predicted_events = extract_events(
        data['videos'], data['frames'], data['predictions'],
        data['probabilities'], runs, frame_step, background_index)
    raw_event_summary, raw_event_rows = event_metrics(
        raw_predicted_events, ground_truth_events, class_names,
        background_index, args.tiou_thresholds)
    processed_event_summary, processed_event_rows = event_metrics(
        predicted_events, ground_truth_events, class_names,
        background_index, args.tiou_thresholds)

    output_dir = args.output_dir
    if output_dir is None:
        stem = os.path.splitext(os.path.basename(args.predictions_file))[0]
        output_dir = os.path.join(
            os.path.dirname(args.predictions_file), 'analysis_{}'.format(stem))
    os.makedirs(output_dir, exist_ok=True)

    report = {
        'source': os.path.abspath(args.predictions_file),
        'frame_step': frame_step,
        'post_processing': {
            'smooth_window': args.smooth_window,
            'merge_gap': args.merge_gap,
            'min_event_length': args.min_event_length,
        },
        'raw_frame_metrics': raw_summary,
        'processed_frame_metrics': processed_summary,
        'raw_event_metrics': raw_event_summary,
        'processed_event_metrics': processed_event_summary,
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    write_csv(os.path.join(output_dir, 'per_class_raw.csv'), raw_per_class)
    write_csv(os.path.join(output_dir, 'per_class_processed.csv'), per_class)
    write_csv(os.path.join(output_dir, 'per_video.csv'), video_rows)
    write_csv(os.path.join(output_dir, 'event_metrics_raw.csv'), raw_event_rows)
    write_csv(
        os.path.join(output_dir, 'event_metrics_processed.csv'), processed_event_rows)
    write_confusion_matrix(
        os.path.join(output_dir, 'confusion_matrix_raw.csv'), raw_matrix, class_names)
    write_confusion_matrix(
        os.path.join(output_dir, 'confusion_matrix_processed.csv'), matrix, class_names)
    write_event_csv(
        os.path.join(output_dir, 'events_raw.csv'),
        ground_truth_events, raw_predicted_events, class_names)
    write_event_csv(
        os.path.join(output_dir, 'events_processed.csv'),
        ground_truth_events, predicted_events, class_names)

    processed_path = os.path.join(output_dir, 'processed_predictions.npz')
    np.savez_compressed(
        processed_path,
        videos=data['videos'],
        frames=data['frames'],
        probabilities=smoothed_probabilities.astype(np.float32),
        labels=data['labels'],
        raw_predictions=data['predictions'],
        predictions=processed_predictions,
        class_names=class_names,
    )

    print('Analysis written to {}'.format(output_dir))
    print('Raw foreground macro F1: {:.4f}'.format(
        raw_summary['foreground_macro_f1']))
    print('Processed foreground macro F1: {:.4f}'.format(
        processed_summary['foreground_macro_f1']))
    for raw_result, processed_result in zip(
            raw_event_summary, processed_event_summary):
        print('tIoU {:.2f}: raw event F1={:.4f}, processed event F1={:.4f}, '
              'raw mAP={:.4f}, processed mAP={:.4f}'.format(
            processed_result['threshold'], raw_result['micro_f1'],
            processed_result['micro_f1'], raw_result['map'],
            processed_result['map']))
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyse predictions exported by evaluate.py --save_predictions.')
    parser.add_argument('predictions_file', help='Path to predictions_*.npz')
    parser.add_argument('--output-dir', default=None, help='Directory for generated reports')
    parser.add_argument('--background-class', default='OTH')
    parser.add_argument('--frame-step', type=int, default=None,
                        help='Frame sampling step; inferred when omitted')
    parser.add_argument('--smooth-window', type=int, default=1,
                        help='Odd moving-average window over class probabilities')
    parser.add_argument('--merge-gap', type=int, default=0,
                        help='Maximum background gap in source frames to merge')
    parser.add_argument('--min-event-length', type=int, default=1,
                        help='Minimum foreground event length in source frames')
    parser.add_argument('--tiou-thresholds', type=float, nargs='+',
                        default=[0.3, 0.5, 0.7])
    return parser.parse_args()


if __name__ == '__main__':
    analyse(parse_args())
