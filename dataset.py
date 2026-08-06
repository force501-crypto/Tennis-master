"""RGB frame and MS-TCN feature datasets for the unified 0006 pipeline."""
from collections import defaultdict
import os

import mxnet as mx
import numpy as np
from mxnet.gluon.data import Dataset


DEFAULT_MODEL_ID = '0006'


def normalize_model_id(model_id):
    """Normalize numeric experiment IDs to the project's four-digit format."""
    value = str(model_id).strip()
    if not value:
        raise ValueError('model_id cannot be empty')
    return value.zfill(4) if value.isdigit() else value


def feature_path(feature_dir, video, frame, chunk_size=1000):
    """Return the stored CNN feature path for one source frame."""
    chunk = int(frame / chunk_size) * chunk_size
    return os.path.join(
        feature_dir, '{}.mp4'.format(video), '{:010d}'.format(chunk),
        '{:010d}.npy'.format(frame))


def image_path(image_dir, video, frame, chunk_size=1000):
    """Return an extracted RGB image path for one source frame."""
    chunk = int(frame / chunk_size) * chunk_size
    return os.path.join(
        image_dir, '{}.mp4'.format(video), '{:010d}'.format(chunk),
        '{:010d}.jpg'.format(frame))


def build_contiguous_runs(samples, frame_step=1):
    """Group ``[video, frame, label]`` samples into contiguous video runs."""
    if frame_step < 1:
        raise ValueError('frame_step must be positive')

    by_video = defaultdict(list)
    for sample in samples:
        by_video[str(sample[0])].append(sample)

    runs = []
    for video in sorted(by_video):
        ordered = sorted(by_video[video], key=lambda sample: int(sample[1]))
        dense_runs = []
        current = []
        previous_frame = None
        for sample in ordered:
            frame = int(sample[1])
            if previous_frame is not None and frame != previous_frame + 1:
                if current:
                    dense_runs.append(current)
                current = []
            current.append(sample)
            previous_frame = frame
        if current:
            dense_runs.append(current)

        for dense_run in dense_runs:
            sampled = dense_run[::frame_step]
            if sampled:
                runs.append(sampled)
    return runs


def load_classes(root):
    path = os.path.join(root, 'classes.names')
    if not os.path.exists(path):
        raise FileNotFoundError('Class list does not exist: {}'.format(path))
    with open(path, 'r') as handle:
        classes = [line.strip() for line in handle if line.strip()]
    if not classes:
        raise ValueError('Class list is empty: {}'.format(path))
    return classes


def load_split_samples(root, split_id, split, classes, video_id=None):
    """Load split membership and frame labels without the legacy dataset."""
    split_path = os.path.join(root, 'splits', split_id, '{}.txt'.format(split))
    if not os.path.exists(split_path):
        raise FileNotFoundError('Dataset split does not exist: {}'.format(split_path))

    requested_video = str(video_id) if video_id else None
    members = []
    with open(split_path, 'r') as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(
                    'Invalid split row {} in {}'.format(line_number, split_path))
            video = fields[0]
            if requested_video and video != requested_video:
                continue
            members.append((video, int(fields[1])))
    if not members:
        suffix = ' for {}'.format(requested_video) if requested_video else ''
        raise ValueError('Split {} contains no samples{}'.format(split, suffix))

    labels_by_video = {}
    for video in sorted({member[0] for member in members}):
        label_path = os.path.join(root, 'annotations', 'labels', '{}.txt'.format(video))
        if not os.path.exists(label_path):
            raise FileNotFoundError('Frame labels do not exist: {}'.format(label_path))
        labels = {}
        with open(label_path, 'r') as handle:
            for line_number, line in enumerate(handle, start=1):
                fields = line.strip().split()
                if not fields:
                    continue
                if len(fields) < 2:
                    raise ValueError(
                        'Invalid label row {} in {}'.format(line_number, label_path))
                class_name = fields[1]
                if class_name not in classes:
                    raise ValueError(
                        'Unknown class {} in {}'.format(class_name, label_path))
                labels[int(fields[0])] = class_name
        labels_by_video[video] = labels

    samples = []
    for video, frame in members:
        try:
            class_name = labels_by_video[video][frame]
        except KeyError:
            raise ValueError(
                'Frame {} in {} has no label'.format(frame, video))
        samples.append([video, frame, class_name])
    return samples


class TennisRGBFrameSet(Dataset):
    """RGB frames used to create model 0006 features."""

    def __init__(self, split, transform, model_id=DEFAULT_MODEL_ID,
                 split_id='02', root='data', video_id=None):
        self.root = root
        self.split = split
        self.model_id = normalize_model_id(model_id)
        self.transform = transform
        self.classes = load_classes(root)
        self.samples = load_split_samples(
            root, split_id, split, self.classes, video_id=video_id)
        self.frames_dir = os.path.join(root, 'frames')
        self.feature_dir = os.path.join(root, 'features', self.model_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video, frame, _ = self.samples[index]
        rgb_path = image_path(self.frames_dir, video, frame)
        if not os.path.exists(rgb_path):
            raise FileNotFoundError('RGB frame does not exist: {}'.format(rgb_path))

        image = mx.image.imread(rgb_path, 1)
        if self.transform is not None:
            image = self.transform(image)
        return image, np.int32(index)

    def feature_output_path(self, index):
        video, frame, _ = self.samples[int(index)]
        return feature_path(self.feature_dir, video, int(frame))

    @property
    def videos(self):
        return sorted({str(sample[0]) for sample in self.samples})

    def __str__(self):
        return 'TennisRGBFrameSet(model_id={}, split={}, videos={}, frames={})'.format(
            self.model_id, self.split, ','.join(self.videos), len(self))


class TennisFeatureSequenceSet(Dataset):
    """Fixed-length MS-TCN sequences backed by model 0006 RGB features.

    Features are read from ``data/features/<model_id>``. Sequences never cross
    a video boundary or a gap in the split file. Short tails are zero padded
    and accompanied by a validity mask.
    """

    def __init__(self, split, model_id=DEFAULT_MODEL_ID, split_id='02',
                 root='data', sequence_length=256, sequence_stride=None,
                 frame_step=1, feature_pool='mean', video_id=None):
        if sequence_length < 2:
            raise ValueError('sequence_length must be at least two')
        if sequence_stride is None:
            sequence_stride = sequence_length
        if sequence_stride < 1:
            raise ValueError('sequence_stride must be positive')
        if sequence_stride > sequence_length:
            raise ValueError(
                'sequence_stride cannot exceed sequence_length because frames '
                'would be skipped')
        if feature_pool not in ('mean', 'flatten'):
            raise ValueError('feature_pool must be mean or flatten')

        self.root = root
        self.split = split
        self.model_id = normalize_model_id(model_id)
        self.video_id = str(video_id) if video_id else None
        self.sequence_length = int(sequence_length)
        self.sequence_stride = int(sequence_stride)
        self.frame_step = int(frame_step)
        self.feature_pool = feature_pool
        self.classes = load_classes(root)
        self.feature_dir = os.path.join(root, 'features', self.model_id)
        self._samples = load_split_samples(
            root, split_id, split, self.classes, video_id=self.video_id)

        self._runs = build_contiguous_runs(
            self._samples, frame_step=self.frame_step)
        self._sequences = []
        for run in self._runs:
            for start in range(0, len(run), self.sequence_stride):
                sequence = run[start:start + self.sequence_length]
                if sequence:
                    self._sequences.append(sequence)
        if not self._sequences:
            raise ValueError('No feature sequences were created for split {}'.format(split))

        first = self._sequences[0][0]
        first_path = feature_path(
            self.feature_dir, first[0], int(first[1]))
        if not os.path.exists(first_path):
            raise FileNotFoundError(
                'Model {} RGB feature does not exist: {}. Run '
                'prepare_features.py before MS-TCN training or evaluation.'.format(
                    self.model_id, first_path))
        first_feature = np.load(first_path)
        self.raw_feature_shape = tuple(first_feature.shape)
        first_feature = self._prepare_feature(first_feature)
        self.feature_size = int(first_feature.size)
        if self.feature_size < 1:
            raise ValueError('Stored feature {} is empty'.format(first_path))

    def __len__(self):
        return len(self._sequences)

    def __getitem__(self, index):
        sequence = self._sequences[index]
        features = np.zeros(
            (self.sequence_length, self.feature_size), dtype=np.float32)
        labels = np.zeros(self.sequence_length, dtype=np.int32)
        mask = np.zeros(self.sequence_length, dtype=np.float32)

        for time_index, sample in enumerate(sequence):
            video, frame, class_name = sample
            path = feature_path(self.feature_dir, video, int(frame))
            if not os.path.exists(path):
                raise FileNotFoundError(
                    'Missing model {} feature: {}'.format(self.model_id, path))
            feature = self._prepare_feature(np.load(path))
            if feature.size != self.feature_size:
                raise ValueError(
                    'Feature size changed from {} to {} at {}'.format(
                        self.feature_size, feature.size, path))
            features[time_index] = feature
            labels[time_index] = self.classes.index(class_name)
            mask[time_index] = 1.0
        return features, labels, mask, np.int32(index)

    def sequence_metadata(self, index):
        sequence = self._sequences[int(index)]
        return [
            (str(sample[0]), int(sample[1]), self.classes.index(sample[2]))
            for sample in sequence
        ]

    def class_counts(self):
        counts = [0] * len(self.classes)
        for sample in self._samples:
            counts[self.classes.index(sample[2])] += 1
        return counts

    @property
    def valid_sample_count(self):
        return sum(len(sequence) for sequence in self._sequences)

    @property
    def videos(self):
        return sorted({str(sample[0]) for sample in self._samples})

    def __str__(self):
        return (
            'TennisFeatureSequenceSet(model_id={}, split={}, videos={}, '
            'sequences={}, valid_frames={}, sequence_length={}, stride={}, '
            'frame_step={}, raw_feature_shape={}, feature_pool={}, '
            'feature_size={})'.format(
                self.model_id, self.split, ','.join(self.videos), len(self),
                self.valid_sample_count, self.sequence_length,
                self.sequence_stride, self.frame_step, self.raw_feature_shape,
                self.feature_pool, self.feature_size))

    def _prepare_feature(self, feature):
        feature = np.asarray(feature, dtype=np.float32)
        if self.feature_pool == 'mean' and feature.ndim >= 3:
            feature = feature.mean(axis=tuple(range(1, feature.ndim)))
        return feature.reshape(-1)
