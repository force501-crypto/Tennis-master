"""Train the unified MS-TCN model 0010 pipeline."""
from absl import app, flags
from absl.flags import FLAGS
import logging
import multiprocessing
import os
import random
import sys
import time

import mxnet as mx
import numpy as np
from mxnet import autograd as ag, gluon

from models.vision.mstcn import MultiStageTemporalConvNet
from dataset import (
    DEFAULT_MODEL_ID, TennisFeatureSequenceSet, normalize_model_id)
from mstcn_utils import (
    checkpoint_epoch, confusion_metrics, final_stage, find_best_checkpoint,
    find_latest_checkpoint, multi_stage_loss, update_confusion)
from utils.losses import get_class_weights


flags.DEFINE_string('model_id', '0010', 'MS-TCN model and feature id.')
flags.DEFINE_string('data_root', 'data', 'Dataset root.')
flags.DEFINE_string('experiment_dir', None,
                    'MS-TCN checkpoint directory; defaults to experiments/0010/mstcn.')
flags.DEFINE_string('split_id', '02', 'Dataset split id.')
flags.DEFINE_string('test_split', 'test_006_full',
                    'Final evaluation split; defaults to the full V006 test split.')
flags.DEFINE_string('video_id', 'V006',
                    'Only this video is used for final evaluation.')
flags.DEFINE_integer('sequence_length', 256, 'Frames per fixed-length MS-TCN sequence.')
flags.DEFINE_integer('train_stride', 128, 'Training sequence stride; smaller values add overlap.')
flags.DEFINE_integer('frame_step', 1, 'Sample every Nth frame inside contiguous split runs.')
flags.DEFINE_enum('feature_pool', 'mean', ['mean', 'flatten'],
                  'Spatially average saved CNN maps or flatten them unchanged.')

flags.DEFINE_integer('mstcn_stages', 3, 'Total prediction and refinement stages.')
flags.DEFINE_integer('mstcn_layers', 7, 'Dilated temporal layers per stage.')
flags.DEFINE_integer('mstcn_channels', 64, 'Hidden temporal feature channels.')
flags.DEFINE_integer('mstcn_kernel_size', 3, 'Odd temporal convolution kernel size.')
flags.DEFINE_float('mstcn_dropout', 0.5, 'Dropout in MS-TCN residual layers.')

flags.DEFINE_integer('batch_size', 4, 'Sequence batch size.')
flags.DEFINE_integer('epochs', 50, 'Total epochs, including resumed epochs.')
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs; zero forces CPU.')
flags.DEFINE_integer('num_workers', -1, 'DataLoader workers; -1 chooses up to eight.')
flags.DEFINE_float('lr', 0.0005, 'Adam learning rate.')
flags.DEFINE_float('wd', 0.0001, 'Weight decay.')
flags.DEFINE_list('lr_steps', '30,40', 'Epochs at which the learning rate is reduced.')
flags.DEFINE_float('lr_factor', 0.5, 'Learning-rate multiplier at each step.')
flags.DEFINE_bool('resume', True, 'Resume from the latest numeric checkpoint when available.')

flags.DEFINE_enum('class_weighting', 'inverse_sqrt',
                  ['none', 'inverse_sqrt', 'effective_num'],
                  'Class weighting used by frame classification loss.')
flags.DEFINE_float('class_weight_beta', 0.9999,
                   'Beta for effective-number class weighting.')
flags.DEFINE_float('focal_gamma', 0.0, 'Focal-loss gamma; zero keeps weighted CE.')
flags.DEFINE_float('smoothing_weight', 0.15,
                   'Weight for adjacent log-probability smoothing.')
flags.DEFINE_float('smoothing_cap', 4.0,
                   'Maximum squared log-probability difference.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('log_interval', 20, 'Mini-batch logging interval.')
flags.DEFINE_integer('max_batches', -1, 'Limit batches per epoch for debugging.')


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

    experiment_dir = FLAGS.experiment_dir or os.path.join(
        'models', 'vision', 'experiments', model_id, 'mstcn')
    os.makedirs(experiment_dir, exist_ok=True)
    _configure_logging(experiment_dir)
    logging.info('Command: %s', ' '.join(sys.argv))
    logging.info('Contexts: %s', contexts)
    logging.info('DataLoader workers: %d', workers)

    train_set = TennisFeatureSequenceSet(
        root=FLAGS.data_root, split='train', model_id=model_id,
        split_id=FLAGS.split_id, sequence_length=FLAGS.sequence_length,
        sequence_stride=FLAGS.train_stride, frame_step=FLAGS.frame_step,
        feature_pool=FLAGS.feature_pool)
    val_set = TennisFeatureSequenceSet(
        root=FLAGS.data_root, split='val', model_id=model_id,
        split_id=FLAGS.split_id, sequence_length=FLAGS.sequence_length,
        sequence_stride=FLAGS.sequence_length, frame_step=FLAGS.frame_step,
        feature_pool=FLAGS.feature_pool)
    test_set = TennisFeatureSequenceSet(
        root=FLAGS.data_root, split=FLAGS.test_split, model_id=model_id,
        video_id=FLAGS.video_id,
        split_id=FLAGS.split_id, sequence_length=FLAGS.sequence_length,
        sequence_stride=FLAGS.sequence_length, frame_step=FLAGS.frame_step,
        feature_pool=FLAGS.feature_pool)
    logging.info('%s', train_set)
    logging.info('%s', val_set)
    logging.info('%s', test_set)

    train_loader = gluon.data.DataLoader(
        train_set, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=workers, last_batch='keep')
    val_loader = gluon.data.DataLoader(
        val_set, batch_size=FLAGS.batch_size, shuffle=False,
        num_workers=workers, last_batch='keep')
    test_loader = gluon.data.DataLoader(
        test_set, batch_size=FLAGS.batch_size, shuffle=False,
        num_workers=workers, last_batch='keep')

    model = MultiStageTemporalConvNet(
        num_classes=len(train_set.classes),
        num_stages=FLAGS.mstcn_stages,
        channels=FLAGS.mstcn_channels,
        num_layers=FLAGS.mstcn_layers,
        kernel_size=FLAGS.mstcn_kernel_size,
        dropout=FLAGS.mstcn_dropout)
    model.initialize(ctx=contexts)
    logging.info('MS-TCN stages=%d, layers=%d, prediction RF=%d, refinement RF=%d frames',
                 model.num_stages, model.num_layers,
                 model.prediction_receptive_field,
                 model.refinement_receptive_field)

    start_epoch = 0
    latest_checkpoint = find_latest_checkpoint(experiment_dir)
    if FLAGS.resume and latest_checkpoint:
        model.load_parameters(latest_checkpoint, ctx=contexts)
        start_epoch = checkpoint_epoch(latest_checkpoint) + 1
        logging.info('Resumed parameters from %s', latest_checkpoint)

    model.hybridize()
    lr_steps = {int(step) for step in FLAGS.lr_steps}
    initial_lr = FLAGS.lr * (
        FLAGS.lr_factor ** sum(step < start_epoch for step in lr_steps))
    trainer = gluon.Trainer(
        model.collect_params(), 'adam',
        {'learning_rate': initial_lr, 'wd': FLAGS.wd})

    class_counts = train_set.class_counts()
    class_weight_values = get_class_weights(
        class_counts, method=FLAGS.class_weighting,
        beta=FLAGS.class_weight_beta)
    class_weights = [
        mx.nd.array(class_weight_values, ctx=context)
        for context in contexts
    ]
    logging.info('Training class counts: %s',
                 dict(zip(train_set.classes, class_counts)))
    logging.info('Class weights: %s', dict(zip(
        train_set.classes,
        [round(float(weight), 4) for weight in class_weight_values])))

    scores_path = os.path.join(experiment_dir, 'scores.txt')
    for epoch in range(start_epoch, FLAGS.epochs):
        if epoch in lr_steps:
            trainer.set_learning_rate(trainer.learning_rate * FLAGS.lr_factor)
        epoch_start = time.time()
        train_result = _train_epoch(
            model, train_loader, train_set.classes, trainer,
            class_weights, contexts, epoch)
        val_result = _evaluate_epoch(
            model, val_loader, val_set.classes, class_weights, contexts)

        checkpoint_path = os.path.join(
            experiment_dir, '{:04d}.params'.format(epoch))
        model.save_parameters(checkpoint_path)
        with open(scores_path, 'a') as score_file:
            score_file.write('{}\t{}\n'.format(
                epoch, val_result['foreground_macro_f1']))

        logging.info(
            '[Epoch %d] train_loss=%.5f, train_fg_f1=%.4f, '
            'val_loss=%.5f, val_accuracy=%.4f, val_fg_f1=%.4f, time=%.1fs',
            epoch, train_result['loss'],
            train_result['foreground_macro_f1'], val_result['loss'],
            val_result['accuracy'], val_result['foreground_macro_f1'],
            time.time() - epoch_start)

    best_checkpoint = find_best_checkpoint(experiment_dir)
    if best_checkpoint is None:
        raise FileNotFoundError(
            'No MS-TCN checkpoint is available in {}'.format(experiment_dir))
    model.load_parameters(best_checkpoint, ctx=contexts)
    test_result = _evaluate_epoch(
        model, test_loader, test_set.classes, class_weights, contexts)
    logging.info('Best checkpoint: %s', best_checkpoint)
    logging.info(
        '[Test] loss=%.5f, accuracy=%.4f, macro_f1=%.4f, foreground_macro_f1=%.4f',
        test_result['loss'], test_result['accuracy'],
        test_result['macro_f1'], test_result['foreground_macro_f1'])


def _train_epoch(model, loader, class_names, trainer, class_weights,
                 contexts, epoch):
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    loss_sum = 0.0
    batches = 0
    epoch_start = time.time()
    for batch_index, batch in enumerate(loader):
        if FLAGS.max_batches > 0 and batch_index >= FLAGS.max_batches:
            break
        data_parts = gluon.utils.split_and_load(
            batch[0], ctx_list=contexts, batch_axis=0, even_split=False)
        label_parts = gluon.utils.split_and_load(
            batch[1], ctx_list=contexts, batch_axis=0, even_split=False)
        mask_parts = gluon.utils.split_and_load(
            batch[2], ctx_list=contexts, batch_axis=0, even_split=False)

        losses = []
        final_logits = []
        with ag.record():
            for data, labels, mask, weights in zip(
                    data_parts, label_parts, mask_parts, class_weights):
                stage_logits = model(data)
                loss, _, _ = multi_stage_loss(
                    stage_logits, labels, mask, weights,
                    gamma=FLAGS.focal_gamma,
                    smoothing_weight=FLAGS.smoothing_weight,
                    smoothing_cap=FLAGS.smoothing_cap)
                losses.append(loss)
                final_logits.append(final_stage(stage_logits))
            ag.backward(losses)
        trainer.step(len(losses))

        batch_loss = float(np.mean([loss.asscalar() for loss in losses]))
        loss_sum += batch_loss
        batches += 1
        for logits, labels, mask in zip(final_logits, label_parts, mask_parts):
            update_confusion(confusion, logits, labels, mask)

        if FLAGS.log_interval > 0 and (batch_index + 1) % FLAGS.log_interval == 0:
            elapsed = max(time.time() - epoch_start, 1e-6)
            frames = int(confusion.sum())
            logging.info(
                '[Epoch %d][Batch %d/%d] loss=%.5f, lr=%.2e, %.1f frames/s',
                epoch, batch_index + 1, len(loader), loss_sum / batches,
                trainer.learning_rate, frames / elapsed)

    metrics, _ = confusion_metrics(confusion, class_names)
    metrics['loss'] = loss_sum / max(batches, 1)
    return metrics


def _evaluate_epoch(model, loader, class_names, class_weights, contexts):
    confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    loss_sum = 0.0
    batches = 0
    for batch_index, batch in enumerate(loader):
        if FLAGS.max_batches > 0 and batch_index >= FLAGS.max_batches:
            break
        data_parts = gluon.utils.split_and_load(
            batch[0], ctx_list=contexts, batch_axis=0, even_split=False)
        label_parts = gluon.utils.split_and_load(
            batch[1], ctx_list=contexts, batch_axis=0, even_split=False)
        mask_parts = gluon.utils.split_and_load(
            batch[2], ctx_list=contexts, batch_axis=0, even_split=False)

        losses = []
        for data, labels, mask, weights in zip(
                data_parts, label_parts, mask_parts, class_weights):
            stage_logits = model(data)
            loss, _, _ = multi_stage_loss(
                stage_logits, labels, mask, weights,
                gamma=FLAGS.focal_gamma,
                smoothing_weight=FLAGS.smoothing_weight,
                smoothing_cap=FLAGS.smoothing_cap)
            losses.append(loss)
            update_confusion(
                confusion, final_stage(stage_logits), labels, mask)
        loss_sum += float(np.mean([loss.asscalar() for loss in losses]))
        batches += 1

    metrics, _ = confusion_metrics(confusion, class_names)
    metrics['loss'] = loss_sum / max(batches, 1)
    return metrics


def _validate_flags():
    if normalize_model_id(FLAGS.model_id) != DEFAULT_MODEL_ID:
        raise ValueError('This unified trainer only supports model 0010')
    if FLAGS.sequence_length < 2:
        raise ValueError('sequence_length must be at least two')
    if FLAGS.train_stride < 1:
        raise ValueError('train_stride must be positive')
    if FLAGS.batch_size < 1:
        raise ValueError('batch_size must be positive')
    if FLAGS.epochs < 0:
        raise ValueError('epochs must be non-negative')
    if FLAGS.num_gpus < 0:
        raise ValueError('num_gpus must be non-negative')


def _configure_logging(experiment_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    log_path = os.path.join(experiment_dir, 'mstcn_log.txt')
    absolute_log_path = os.path.abspath(log_path)
    if not any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, 'baseFilename', None) == absolute_log_path
            for handler in logger.handlers):
        logger.addHandler(logging.FileHandler(log_path))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
