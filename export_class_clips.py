"""Export foreground prediction clips for one source video.

The input is the NPZ file produced by ``evaluate.py --save_predictions`` or
``analyze_predictions.py``. Only the requested video is processed. The
background class (OTH by default) is never written to an output video.
"""
import argparse
import csv
import os
import re

import cv2
import numpy as np

from analyze_predictions import (
    contiguous_runs,
    infer_frame_step,
    label_segments,
    load_predictions,
)


def safe_filename(value):
    """Return a filesystem-safe class name."""
    cleaned = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('._')
    return cleaned or 'class'


def select_video(data, video_id):
    """Select one video while preserving the frame-sorted NPZ arrays."""
    selection = data['videos'] == video_id
    if not np.any(selection):
        available = ', '.join(sorted(np.unique(data['videos']).tolist()))
        raise ValueError(
            'Video {!r} is not present in the prediction file. Available: {}'.format(
                video_id, available or '(none)'))

    return {
        'videos': data['videos'][selection],
        'frames': data['frames'][selection],
        'predictions': data['predictions'][selection],
        'probabilities': data['probabilities'][selection],
        'class_names': data['class_names'],
    }


def build_events(data, background_class, frame_step=None, min_event_frames=1):
    """Convert frame predictions into foreground-only temporal events."""
    class_names = data['class_names']
    background_matches = np.where(class_names == background_class)[0]
    if len(background_matches) == 0:
        raise ValueError(
            'Background class {!r} is not in {}'.format(
                background_class, class_names.tolist()))
    background_index = int(background_matches[0])

    predictions = data['predictions']
    if np.any(predictions < 0) or np.any(predictions >= len(class_names)):
        raise ValueError('predictions contain an invalid class index')

    if frame_step is None:
        frame_step = infer_frame_step(data['videos'], data['frames'])
    if frame_step < 1:
        raise ValueError('frame_step must be at least 1')
    if min_event_frames < 1:
        raise ValueError('min_event_frames must be at least 1')

    runs = contiguous_runs(data['videos'], data['frames'], frame_step)
    events = []
    class_counts = {}
    for run_start, run_end in runs:
        for start, end, class_index in label_segments(
                predictions, run_start, run_end):
            if class_index == background_index:
                continue
            duration = (end - start) * frame_step
            if duration < min_event_frames:
                continue

            class_name = str(class_names[class_index])
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            events.append({
                'class_index': class_index,
                'class_name': class_name,
                'event_index': class_counts[class_name],
                'start_frame': int(data['frames'][start]),
                'end_frame': int(data['frames'][end - 1] + frame_step - 1),
                'sample_count': int(end - start),
                'score': float(np.mean(
                    data['probabilities'][start:end, class_index])),
            })
    return events, frame_step


def open_source(video_path):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError('Cannot open source video: {}'.format(video_path))

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError('Source video has invalid metadata: {}'.format(video_path))
    return capture, fps, (width, height), frame_count


def open_writer(path, fps, frame_size, codec):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*codec), fps, frame_size)
    if not writer.isOpened():
        writer.release()
        raise IOError(
            'Cannot create output video {} with codec {}'.format(path, codec))
    return writer


def event_output_path(output_dir, event, output_mode):
    class_name = safe_filename(event['class_name'])
    if output_mode == 'per_class':
        return os.path.join(output_dir, '{}.mp4'.format(class_name))
    filename = '{}_{:04d}_{:010d}_{:010d}.mp4'.format(
        class_name, event['event_index'], event['start_frame'],
        event['end_frame'])
    return os.path.join(output_dir, class_name, filename)


def write_event(capture, writer, start_frame, end_frame):
    """Write an inclusive source-frame interval and return frames written."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for _ in range(start_frame, end_frame + 1):
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        written += 1
    return written


def export_events(video_path, output_dir, events, output_mode='per_class',
                  before_seconds=1.0, after_seconds=3.0, codec='mp4v'):
    """Cut events, keeping every foreground class in a separate output."""
    if before_seconds < 0 or after_seconds < 0:
        raise ValueError('clip context seconds cannot be negative')
    if len(codec) != 4:
        raise ValueError('codec must contain exactly four characters')

    os.makedirs(output_dir, exist_ok=True)
    capture, fps, frame_size, frame_count = open_source(video_path)
    before_frames = int(round(before_seconds * fps))
    after_frames = int(round(after_seconds * fps))
    writers = {}
    manifest = []
    covered_until = -1
    skipped_covered_events = 0
    try:
        for event in sorted(events, key=lambda item: item['start_frame']):
            source_start = event['start_frame']
            source_end = event['end_frame']
            # The preceding accepted clip already contains this event's start
            # in its trailing context, so exporting it again would duplicate
            # footage. This check is global across all foreground classes.
            if source_start <= covered_until:
                skipped_covered_events += 1
                continue
            # Export [event_start - 1s, event_end + 3s] by default.
            write_start = max(0, source_start - before_frames)
            write_end = min(frame_count - 1, source_end + after_frames)
            if write_start > write_end:
                continue

            output_path = event_output_path(output_dir, event, output_mode)
            if output_mode == 'per_class':
                writer = writers.get(event['class_name'])
                if writer is None:
                    writer = open_writer(output_path, fps, frame_size, codec)
                    writers[event['class_name']] = writer
            else:
                writer = open_writer(output_path, fps, frame_size, codec)

            written = write_event(
                capture, writer, write_start, write_end)
            if output_mode == 'per_event':
                writer.release()
            if written == 0:
                raise IOError(
                    'No frames could be read for event {} {}-{}'.format(
                        event['class_name'], write_start, write_end))

            row = dict(event)
            row.update({
                'written_start_frame': write_start,
                'written_end_frame': write_start + written - 1,
                'written_frames': written,
                'output_file': os.path.relpath(output_path, output_dir),
            })
            manifest.append(row)
            covered_until = write_end
    finally:
        capture.release()
        for writer in writers.values():
            writer.release()

    return manifest, fps, frame_count, skipped_covered_events


def write_manifest(path, video_id, rows):
    fields = [
        'video_id', 'class_name', 'event_index', 'start_frame', 'end_frame',
        'written_start_frame', 'written_end_frame', 'sample_count',
        'written_frames', 'score', 'output_file',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {key: row[key] for key in fields if key != 'video_id'}
            output['video_id'] = video_id
            writer.writerow(output)


def export_video_clips(predictions_file, video_id='V006', data_root='data',
                       video_file=None, output_dir=None,
                       background_class='OTH', output_mode='per_class',
                       frame_step=None, min_event_frames=1,
                       before_seconds=1.0, after_seconds=3.0, codec='mp4v'):
    """Export one video's foreground clips for CLI and evaluate.py callers."""
    data = select_video(
        load_predictions(predictions_file, allow_unlabeled=True), video_id)
    events, frame_step = build_events(
        data, background_class, frame_step, min_event_frames)
    if not events:
        raise ValueError(
            'No non-{} events remain for {}'.format(
                background_class, video_id))

    video_path = video_file or os.path.join(
        data_root, 'videos', '{}.mp4'.format(video_id))
    output_dir = output_dir or os.path.join(
        os.path.dirname(os.path.abspath(predictions_file)),
        'class_clips', video_id)
    manifest, fps, source_frames, skipped_covered_events = export_events(
        video_path=os.path.abspath(video_path),
        output_dir=os.path.abspath(output_dir),
        events=events,
        output_mode=output_mode,
        before_seconds=before_seconds,
        after_seconds=after_seconds,
        codec=codec,
    )
    manifest_path = os.path.join(os.path.abspath(output_dir), 'manifest.csv')
    write_manifest(manifest_path, video_id, manifest)

    class_names = sorted({event['class_name'] for event in manifest})
    print('Video: {}'.format(video_id))
    print('Source: {}'.format(os.path.abspath(video_path)))
    print('Prediction frame step: {}'.format(frame_step))
    print('Source FPS / frames: {:.3f} / {}'.format(fps, source_frames))
    print('Clip context: t-{:.3g}s to t+{:.3g}s'.format(
        before_seconds, after_seconds))
    print('Exported {} events into {} separate class(es): {}'.format(
        len(manifest), len(class_names), ', '.join(class_names)))
    print('Skipped {} event(s) already covered by the previous clip'.format(
        skipped_covered_events))
    print('OTH/background output: disabled')
    print('Output: {}'.format(os.path.abspath(output_dir)))
    print('Manifest: {}'.format(manifest_path))
    print('Note: OpenCV output preserves video frames but does not copy audio.')
    return manifest


def export(args):
    return export_video_clips(
        predictions_file=args.predictions_file,
        video_id=args.video_id,
        data_root=args.data_root,
        video_file=args.video_file,
        output_dir=args.output_dir,
        background_class=args.background_class,
        output_mode=args.output_mode,
        frame_step=args.frame_step,
        min_event_frames=args.min_event_frames,
        before_seconds=args.before_seconds,
        after_seconds=args.after_seconds,
        codec=args.codec,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Cut non-OTH predictions from one video into separate class videos.'))
    parser.add_argument(
        'predictions_file',
        help='NPZ from evaluate.py or processed_predictions.npz')
    parser.add_argument('--video-id', default='V006',
                        help='Only this video is processed (default: V006)')
    parser.add_argument('--data-root', default='data',
                        help='Dataset root containing videos/ (default: data)')
    parser.add_argument('--video-file', default=None,
                        help='Explicit source MP4 path; overrides --data-root')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: beside prediction NPZ)')
    parser.add_argument('--background-class', default='OTH',
                        help='Class to exclude completely (default: OTH)')
    parser.add_argument('--output-mode', choices=['per_class', 'per_event'],
                        default='per_class',
                        help='One MP4 per class or per event (default: per_class)')
    parser.add_argument('--frame-step', type=int, default=None,
                        help='Prediction sampling step; inferred when omitted')
    parser.add_argument('--min-event-frames', type=int, default=1,
                        help='Discard shorter foreground events (source frames)')
    parser.add_argument('--before-seconds', type=float, default=1.0,
                        help='Seconds before each event (default: 1)')
    parser.add_argument('--after-seconds', type=float, default=3.0,
                        help='Seconds after each event (default: 3)')
    parser.add_argument('--codec', default='mp4v',
                        help='FourCC output codec (default: mp4v)')
    return parser.parse_args()


if __name__ == '__main__':
    export(parse_args())
