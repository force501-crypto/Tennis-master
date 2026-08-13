"""Run the author's original model 0006 frame classifier on a full video."""
import argparse
import multiprocessing
import os
import time

os.environ.setdefault('MXNET_CUDNN_AUTOTUNE_DEFAULT', '0')

import mxnet as mx
import numpy as np
from gluoncv.model_zoo import get_model
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from tqdm import tqdm

from dataset import DEFAULT_MODEL_ID, TennisRGBFrameSet, normalize_model_id
from export_class_clips import export_video_clips
from models.vision.definitions import FrameModel
from mstcn_utils import find_best_checkpoint


def build_model(class_count, backbone, data_shape, contexts):
    network = FrameModel(
        get_model(backbone, pretrained=False).features, class_count)
    network.initialize(ctx=contexts)
    network(mx.nd.zeros(
        (1, 3, data_shape, data_shape), ctx=contexts[0]))
    return network


def predict(model, loader, dataset, contexts):
    records = []
    for images, indices in tqdm(loader, desc='Original 0006 frame inference'):
        image_parts = gluon.utils.split_and_load(
            images, ctx_list=contexts, batch_axis=0, even_split=False)
        index_parts = gluon.utils.split_and_load(
            indices, ctx_list=contexts, batch_axis=0, even_split=False)
        for image_part, index_part in zip(image_parts, index_parts):
            logits = model(image_part).asnumpy().astype(np.float32)
            for index, row in zip(
                    index_part.asnumpy().astype(np.int64), logits):
                video, frame, _ = dataset.samples[int(index)]
                records.append((str(video), int(frame), row))
    if not records:
        raise ValueError('Original model 0006 produced no predictions')
    records.sort(key=lambda item: (item[0], item[1]))
    return records


def save_predictions(path, records, class_names):
    logits = np.stack([record[2] for record in records])
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1).astype(np.int32)
    videos = np.asarray([record[0] for record in records])
    frames = np.asarray([record[1] for record in records], dtype=np.int64)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(
        path,
        paths=np.asarray([
            '{}:{:010d}'.format(video, frame)
            for video, frame in zip(videos, frames)]),
        videos=videos,
        frames=frames,
        logits=logits,
        probabilities=probabilities.astype(np.float32),
        labels=np.full(len(frames), -1, dtype=np.int32),
        predictions=predictions,
        class_names=np.asarray(class_names))
    return predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a full video with the original 0006 DenseNet model.')
    parser.add_argument('--video-id', default='test001',
                        help='Video ID without .mp4')
    parser.add_argument('--model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--backbone', default='DenseNet121')
    parser.add_argument('--params-file', default=None)
    parser.add_argument('--data-shape', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=-1)
    parser.add_argument('--predictions-file', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--smooth-window', type=int, default=9)
    parser.add_argument('--min-event-frames', type=int, default=5)
    parser.add_argument('--min-event-confidence', type=float, default=0.50)
    parser.add_argument('--before-seconds', type=float, default=1.0)
    parser.add_argument('--after-seconds', type=float, default=3.0)
    args = parser.parse_args()
    if normalize_model_id(args.model_id) != DEFAULT_MODEL_ID:
        parser.error('this script only supports original model 0006')
    if not args.video_id or args.video_id.lower().endswith('.mp4'):
        parser.error('--video-id must not include .mp4')
    if args.batch_size < 1 or args.num_gpus < 0:
        parser.error('invalid batch size or GPU count')
    return args


def main():
    args = parse_args()
    model_id = normalize_model_id(args.model_id)
    workers = args.num_workers
    if workers < 0:
        workers = min(8, multiprocessing.cpu_count())
    contexts = ([mx.gpu(i) for i in range(args.num_gpus)]
                if args.num_gpus else [mx.cpu()])

    transform = transforms.Compose([
        transforms.Resize(args.data_shape + 32),
        transforms.CenterCrop(args.data_shape),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = TennisRGBFrameSet(
        root=args.data_root, split='full_video', video_id=args.video_id,
        full_video=True, model_id=model_id, transform=transform)
    loader = gluon.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=workers, last_batch='keep')

    model = build_model(
        len(dataset.classes), args.backbone, args.data_shape, contexts)
    experiment_dir = os.path.join(
        'models', 'vision', 'experiments', model_id)
    checkpoint = args.params_file or find_best_checkpoint(experiment_dir)
    if checkpoint is None or not os.path.isfile(checkpoint):
        raise FileNotFoundError('No original 0006 checkpoint found')
    model.load_parameters(checkpoint, ctx=contexts)
    model.hybridize()
    print(dataset)
    print('Loaded original frame model: {}'.format(checkpoint))
    print('Contexts: {}'.format(contexts))

    started = time.time()
    records = predict(model, loader, dataset, contexts)
    predictions_file = args.predictions_file or os.path.join(
        experiment_dir, 'predictions_{}_original.npz'.format(args.video_id))
    predictions = save_predictions(
        predictions_file, records, dataset.classes)
    print('Predicted {} frames in {:.1f}s'.format(
        len(records), time.time() - started))
    for index, class_name in enumerate(dataset.classes):
        print('{}: {}'.format(class_name, int((predictions == index).sum())))

    output_dir = args.output_dir or os.path.join(
        experiment_dir, 'original_class_clips',
        '{}_original_output'.format(args.video_id))
    export_video_clips(
        predictions_file=predictions_file,
        video_id=args.video_id,
        data_root=args.data_root,
        output_dir=output_dir,
        min_event_frames=args.min_event_frames,
        smooth_window=args.smooth_window,
        min_event_confidence=args.min_event_confidence,
        before_seconds=args.before_seconds,
        after_seconds=args.after_seconds)


if __name__ == '__main__':
    main()
