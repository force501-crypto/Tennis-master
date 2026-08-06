"""Evaluate MS-TCN model 0010 on V006 and export class-separated clips."""
from absl import app, flags
from absl.flags import FLAGS
import multiprocessing
import os
import random
import time

import mxnet as mx
import numpy as np
from mxnet import gluon

from models.vision.mstcn import MultiStageTemporalConvNet
from dataset import (
    DEFAULT_MODEL_ID, TennisFeatureSequenceSet, normalize_model_id)
from export_class_clips import export_video_clips
from mstcn_utils import confusion_metrics, final_stage, find_best_checkpoint


flags.DEFINE_string('model_id', '0010', 'MS-TCN model and feature id.')
flags.DEFINE_string('data_root', 'data', 'Dataset root.')
flags.DEFINE_string('split_id', '02', 'Dataset split id.')
flags.DEFINE_string('split', 'test_006_full',
                    'V006 split to evaluate (default: test_006_full).')
flags.DEFINE_string('video_id', 'V006',
                    'Only this video is evaluated and clipped.')
flags.DEFINE_integer('sequence_length', 256, 'Frames per evaluation sequence.')
flags.DEFINE_integer('sequence_stride', 256,
                     'Evaluation stride; overlap is averaged when smaller than length.')
flags.DEFINE_integer('frame_step', 1, 'Sample every Nth frame inside contiguous runs.')
flags.DEFINE_enum('feature_pool', 'mean', ['mean', 'flatten'],
                  'Must match the spatial feature pooling used during training.')

flags.DEFINE_integer('mstcn_stages', 3, 'Total prediction and refinement stages.')
flags.DEFINE_integer('mstcn_layers', 7, 'Dilated temporal layers per stage.')
flags.DEFINE_integer('mstcn_channels', 64, 'Hidden temporal feature channels.')
flags.DEFINE_integer('mstcn_kernel_size', 3, 'Odd temporal convolution kernel size.')
flags.DEFINE_float('mstcn_dropout', 0.5, 'Dropout used during training; disabled in evaluation.')

flags.DEFINE_integer('batch_size', 4, 'Sequence batch size.')
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs; zero forces CPU.')
flags.DEFINE_integer('num_workers', -1, 'DataLoader workers; -1 chooses up to eight.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string('params_file', None,
                    'MS-TCN checkpoint; defaults to experiments/0010/mstcn.')
flags.DEFINE_bool('save_predictions', True,
                  'Write an analyze_predictions.py compatible NPZ file.')
flags.DEFINE_string('predictions_file', None, 'Optional prediction output path.')
flags.DEFINE_bool('compress_predictions', True,
                  'Compress NPZ output; disable for faster mounted-drive writes.')
flags.DEFINE_bool('export_clips', True,
                  'After evaluation, remove OTH and export separate class MP4 files.')
flags.DEFINE_string('video_file', None,
                    'Source V006 MP4; defaults to data/videos/V006.mp4.')
flags.DEFINE_string('clips_output_dir', None,
                    'Clip directory; defaults to experiment/0010/mstcn/class_clips/V006.')
flags.DEFINE_enum('clip_output_mode', 'per_class', ['per_class', 'per_event'],
                  'Write one MP4 per class or one MP4 per detected event.')
flags.DEFINE_string('background_class', 'OTH', 'Class excluded from clips.')
flags.DEFINE_integer('min_event_frames', 1,
                     'Discard foreground events shorter than this many frames.')
flags.DEFINE_integer('padding_frames', 0,
                     'Context frames added around exported events.')
flags.DEFINE_string('clip_codec', 'mp4v', 'FourCC codec for output MP4 files.')


def main(_argv):
    _validate_flags()
    model_id = normalize_model_id(FLAGS.model_id)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    mx.random.seed(FLAGS.seed)

    workers = FLAGS.num_workers
    if workers < 0:
        workers = min(8, multiprocessing.cpu_count())
    contexts = (
        [mx.gpu(index) for index in range(FLAGS.num_gpus)]
        if FLAGS.num_gpus > 0 else [mx.cpu()])

    dataset = TennisFeatureSequenceSet(
        root=FLAGS.data_root, split=FLAGS.split, model_id=model_id,
        video_id=FLAGS.video_id,
        split_id=FLAGS.split_id, sequence_length=FLAGS.sequence_length,
        sequence_stride=FLAGS.sequence_stride, frame_step=FLAGS.frame_step,
        feature_pool=FLAGS.feature_pool)
    if dataset.videos != [FLAGS.video_id]:
        raise ValueError(
            'Evaluation must contain only {} but found {}'.format(
                FLAGS.video_id, dataset.videos))
    loader = gluon.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=False,
        num_workers=workers, last_batch='keep')
    print(dataset)
    print('Evaluation contexts: {}'.format(contexts))
    print('Evaluation DataLoader workers: {}'.format(workers))

    model = MultiStageTemporalConvNet(
        num_classes=len(dataset.classes),
        num_stages=FLAGS.mstcn_stages,
        channels=FLAGS.mstcn_channels,
        num_layers=FLAGS.mstcn_layers,
        kernel_size=FLAGS.mstcn_kernel_size,
        dropout=FLAGS.mstcn_dropout)
    model.initialize(ctx=contexts)

    experiment_dir = os.path.join(
        'models', 'vision', 'experiments', model_id, 'mstcn')
    checkpoint = FLAGS.params_file or find_best_checkpoint(experiment_dir)
    if checkpoint is None or not os.path.exists(checkpoint):
        raise FileNotFoundError(
            'No MS-TCN checkpoint found under {}'.format(experiment_dir))
    model.load_parameters(checkpoint, ctx=contexts)
    model.hybridize()
    print('Loaded MS-TCN parameters: {}'.format(checkpoint))

    start = time.time()
    records = _collect_predictions(model, loader, dataset, contexts)
    elapsed = time.time() - start
    arrays = _records_to_arrays(records, dataset.classes)

    confusion = np.zeros(
        (len(dataset.classes), len(dataset.classes)), dtype=np.int64)
    np.add.at(confusion, (arrays['labels'], arrays['predictions']), 1)
    metrics, per_class = confusion_metrics(confusion, dataset.classes)
    print('Confusion matrix (ground truth rows, prediction columns):')
    print(confusion)
    print('Accuracy={:.4f}, Macro F1={:.4f}, Foreground Macro F1={:.4f}'.format(
        metrics['accuracy'], metrics['macro_f1'],
        metrics['foreground_macro_f1']))
    for row in per_class:
        print('{class_name}: precision={precision:.4f}, recall={recall:.4f}, '
              'f1={f1:.4f}, support={support}'.format(**row))
    print('Evaluated {} unique frames in {:.1f} seconds'.format(
        len(arrays['frames']), elapsed))

    if FLAGS.save_predictions or FLAGS.predictions_file:
        output_path = FLAGS.predictions_file
        if output_path is None:
            output_path = os.path.join(
                experiment_dir, 'predictions_{}.npz'.format(FLAGS.video_id))
        _save_predictions(output_path, arrays, FLAGS.compress_predictions)
        print('Saved {} predictions to {} ({})'.format(
            len(arrays['frames']), output_path,
            'compressed' if FLAGS.compress_predictions else 'uncompressed'))

        if FLAGS.export_clips:
            clips_output_dir = FLAGS.clips_output_dir or os.path.join(
                experiment_dir, 'class_clips', FLAGS.video_id)
            export_video_clips(
                predictions_file=output_path,
                video_id=FLAGS.video_id,
                data_root=FLAGS.data_root,
                video_file=FLAGS.video_file,
                output_dir=clips_output_dir,
                background_class=FLAGS.background_class,
                output_mode=FLAGS.clip_output_mode,
                frame_step=FLAGS.frame_step,
                min_event_frames=FLAGS.min_event_frames,
                padding_frames=FLAGS.padding_frames,
                codec=FLAGS.clip_codec,
            )


def _collect_predictions(model, loader, dataset, contexts):
    records = {}
    for batch in loader:
        data_parts = gluon.utils.split_and_load(
            batch[0], ctx_list=contexts, batch_axis=0, even_split=False)
        mask_parts = gluon.utils.split_and_load(
            batch[2], ctx_list=contexts, batch_axis=0, even_split=False)
        index_parts = gluon.utils.split_and_load(
            batch[3], ctx_list=contexts, batch_axis=0, even_split=False)

        for data, mask, sequence_indices in zip(
                data_parts, mask_parts, index_parts):
            stage_logits = model(data)
            logits_ntc = final_stage(stage_logits).transpose(
                (0, 2, 1)).asnumpy()
            mask_np = mask.asnumpy() > 0.5
            indices_np = sequence_indices.asnumpy().astype(np.int32)

            for batch_index, sequence_index in enumerate(indices_np):
                metadata = dataset.sequence_metadata(sequence_index)
                valid_count = int(mask_np[batch_index].sum())
                if valid_count != len(metadata):
                    raise ValueError(
                        'Mask/metadata mismatch for sequence {}'.format(sequence_index))
                for time_index, (video, frame, label) in enumerate(metadata):
                    key = (video, frame)
                    logits = logits_ntc[batch_index, time_index].astype(
                        np.float64, copy=False)
                    if key not in records:
                        records[key] = {
                            'logit_sum': logits.copy(),
                            'count': 1,
                            'label': label,
                        }
                    else:
                        if records[key]['label'] != label:
                            raise ValueError('Conflicting labels for {} frame {}'.format(
                                video, frame))
                        records[key]['logit_sum'] += logits
                        records[key]['count'] += 1
    if not records:
        raise ValueError('MS-TCN evaluation produced no predictions')
    return records


def _records_to_arrays(records, class_names):
    keys = sorted(records, key=lambda item: (item[0], item[1]))
    logits = np.stack([
        records[key]['logit_sum'] / records[key]['count']
        for key in keys
    ]).astype(np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1).astype(np.int32)
    videos = np.asarray([key[0] for key in keys])
    frames = np.asarray([key[1] for key in keys], dtype=np.int64)
    labels = np.asarray(
        [records[key]['label'] for key in keys], dtype=np.int32)
    paths = np.asarray([
        '{}:{:010d}'.format(video, frame)
        for video, frame in keys
    ])
    return {
        'paths': paths,
        'videos': videos,
        'frames': frames,
        'logits': logits,
        'probabilities': probabilities.astype(np.float32),
        'labels': labels,
        'predictions': predictions,
        'class_names': np.asarray(class_names),
    }


def _save_predictions(output_path, arrays, compressed):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_npz = np.savez_compressed if compressed else np.savez
    save_npz(output_path, **arrays)


def _validate_flags():
    if normalize_model_id(FLAGS.model_id) != DEFAULT_MODEL_ID:
        raise ValueError('This unified evaluator only supports model 0010')
    if FLAGS.sequence_length < 2:
        raise ValueError('sequence_length must be at least two')
    if FLAGS.sequence_stride < 1:
        raise ValueError('sequence_stride must be positive')
    if FLAGS.batch_size < 1:
        raise ValueError('batch_size must be positive')
    if FLAGS.num_gpus < 0:
        raise ValueError('num_gpus must be non-negative')
    if FLAGS.video_id != 'V006':
        raise ValueError('This unified evaluator is intentionally restricted to V006')
    if FLAGS.export_clips and not (FLAGS.save_predictions or FLAGS.predictions_file):
        raise ValueError('Clip export requires prediction saving')
    if FLAGS.min_event_frames < 1:
        raise ValueError('min_event_frames must be positive')
    if FLAGS.padding_frames < 0:
        raise ValueError('padding_frames cannot be negative')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
