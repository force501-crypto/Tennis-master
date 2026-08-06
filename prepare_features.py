"""Extract RGB + optical-flow features from frame model 0010 for MS-TCN."""
import os

os.environ.setdefault('MXNET_CUDNN_AUTOTUNE_DEFAULT', '0')

from absl import app, flags
from absl.flags import FLAGS
import multiprocessing

import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import get_model
from tqdm import tqdm

from dataset import (
    DEFAULT_MODEL_ID, TennisTwoStreamFrameSet, normalize_model_id)
from models.vision.definitions import TwoStreamModel
from mstcn_utils import find_best_checkpoint
from utils.transforms import TwoStreamNormalize


flags.DEFINE_string('model_id', '0010', 'Frame model and feature id.')
flags.DEFINE_list(
    'splits', 'train,val,test_006_full',
    'Splits whose model 0010 features are generated.')
flags.DEFINE_string('split_id', '02', 'Dataset split id.')
flags.DEFINE_string('data_root', 'data', 'Dataset root.')
flags.DEFINE_string('backbone', 'DenseNet121', 'Frame model backbone.')
flags.DEFINE_string('params_file', None,
                    'Frame checkpoint; defaults to best checkpoint in 0010 root.')
flags.DEFINE_integer('data_shape', 512, 'Input center-crop size.')
flags.DEFINE_integer('batch_size', 8, 'Feature extraction batch size.')
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs; zero forces CPU.')
flags.DEFINE_integer('num_workers', -1, 'DataLoader workers; -1 chooses up to eight.')
flags.DEFINE_bool('overwrite', False, 'Replace existing feature arrays.')


def main(_argv):
    _validate_flags()
    model_id = normalize_model_id(FLAGS.model_id)
    workers = FLAGS.num_workers
    if workers < 0:
        workers = min(8, multiprocessing.cpu_count())
    contexts = (
        [mx.gpu(index) for index in range(FLAGS.num_gpus)]
        if FLAGS.num_gpus > 0 else [mx.cpu()])

    transform = transforms.Compose([
        transforms.Resize(FLAGS.data_shape + 32),
        transforms.CenterCrop(FLAGS.data_shape),
        TwoStreamNormalize(),
    ])
    datasets = [
        TennisTwoStreamFrameSet(
            root=FLAGS.data_root, split=split, split_id=FLAGS.split_id,
            model_id=model_id, transform=transform)
        for split in FLAGS.splits
    ]
    classes = datasets[0].classes
    if any(dataset.classes != classes for dataset in datasets[1:]):
        raise ValueError('Class definitions differ between feature splits')

    model = _build_frame_model(len(classes), contexts)
    frame_experiment_dir = os.path.join(
        'models', 'vision', 'experiments', model_id)
    checkpoint = FLAGS.params_file or find_best_checkpoint(frame_experiment_dir)
    if checkpoint is None or not os.path.exists(checkpoint):
        raise FileNotFoundError(
            'No frame checkpoint found for model {}'.format(model_id))
    model.load_parameters(checkpoint, ctx=contexts)
    model.hybridize()
    print('Loaded model {} two-stream frame parameters: {}'.format(
        model_id, checkpoint))
    print('Feature output root: {}'.format(os.path.join(
        FLAGS.data_root, 'features', model_id)))

    total_saved = 0
    total_skipped = 0
    for dataset in datasets:
        print(dataset)
        loader = gluon.data.DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=False,
            num_workers=workers, last_batch='keep')
        saved, skipped = _extract_split(
            model, loader, dataset, contexts, FLAGS.overwrite)
        total_saved += saved
        total_skipped += skipped
        print('{}: saved={}, existing={}'.format(
            dataset.split, saved, skipped))
    print('Feature extraction complete: saved={}, existing={}'.format(
        total_saved, total_skipped))


def _build_frame_model(class_count, contexts):
    rgb_backbone = get_model(FLAGS.backbone, pretrained=False).features
    flow_backbone = get_model(FLAGS.backbone, pretrained=False).features
    model = TwoStreamModel(rgb_backbone, flow_backbone, class_count)
    model.initialize(ctx=contexts)
    dummy = mx.nd.zeros(
        (1, 6, FLAGS.data_shape, FLAGS.data_shape), ctx=contexts[0])
    model(dummy)
    return model


def _pooled_two_stream_features(model, data):
    rgb = mx.nd.slice_axis(data, axis=1, begin=0, end=3)
    flow = mx.nd.slice_axis(data, axis=1, begin=3, end=6)
    rgb_features = model.features_rgb(rgb)
    flow_features = model.features_flow(flow)
    if len(rgb_features.shape) > 2:
        axes = tuple(range(2, len(rgb_features.shape)))
        rgb_features = rgb_features.mean(axis=axes)
        flow_features = flow_features.mean(axis=axes)
    rgb_features = rgb_features.reshape((0, -1))
    flow_features = flow_features.reshape((0, -1))
    return mx.nd.concat(rgb_features, flow_features, dim=1)


def _extract_split(model, loader, dataset, contexts, overwrite):
    saved = 0
    skipped = 0
    for batch in tqdm(loader, desc='Extracting {}'.format(dataset.split)):
        indices = batch[1].asnumpy().astype(np.int64)
        output_paths = [dataset.feature_output_path(index) for index in indices]
        if not overwrite and all(os.path.exists(path) for path in output_paths):
            skipped += len(output_paths)
            continue

        data_parts = gluon.utils.split_and_load(
            batch[0], ctx_list=contexts, batch_axis=0, even_split=False)
        index_parts = gluon.utils.split_and_load(
            batch[1], ctx_list=contexts, batch_axis=0, even_split=False)
        for data, part_indices in zip(data_parts, index_parts):
            features = _pooled_two_stream_features(model, data).asnumpy()
            part_indices = part_indices.asnumpy().astype(np.int64)
            for feature, index in zip(features, part_indices):
                path = dataset.feature_output_path(index)
                if os.path.exists(path) and not overwrite:
                    skipped += 1
                    continue
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.save(path, feature.astype(np.float32, copy=False))
                saved += 1
    return saved, skipped


def _validate_flags():
    if normalize_model_id(FLAGS.model_id) != DEFAULT_MODEL_ID:
        raise ValueError('Feature extraction only supports model 0010')
    if not FLAGS.splits:
        raise ValueError('At least one split is required')
    if FLAGS.data_shape < 32:
        raise ValueError('data_shape must be at least 32')
    if FLAGS.batch_size < 1:
        raise ValueError('batch_size must be positive')
    if FLAGS.num_gpus < 0:
        raise ValueError('num_gpus must be non-negative')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
