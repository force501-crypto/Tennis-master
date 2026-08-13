"""Extract RGB frames from one or more videos for the model 0006 pipeline."""
import argparse
import os

from utils.video import video_to_frames


def normalize_video_id(value):
    """Return a video identifier without its .mp4 suffix."""
    video_id = str(value).strip()
    if video_id.lower().endswith('.mp4'):
        video_id = video_id[:-4]
    if not video_id:
        raise ValueError('video id cannot be empty')
    return video_id


def extract_videos(video_ids, data_root='data', overwrite=False,
                   every=1, chunk_size=1000):
    videos_dir = os.path.join(data_root, 'videos')
    frames_dir = os.path.join(data_root, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    for value in video_ids:
        video_id = normalize_video_id(value)
        video_path = os.path.join(videos_dir, '{}.mp4'.format(video_id))
        if not os.path.isfile(video_path):
            raise FileNotFoundError('Source video does not exist: {}'.format(
                os.path.abspath(video_path)))
        print('Extracting {} into {}'.format(
            os.path.abspath(video_path), os.path.abspath(frames_dir)))
        output = video_to_frames(
            video_path=video_path,
            frames_dir=frames_dir,
            overwrite=overwrite,
            every=every,
            chunk_size=chunk_size)
        if output is None:
            raise RuntimeError('Frame extraction failed for {}'.format(video_path))
        print('Frames ready: {}'.format(os.path.abspath(output)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract RGB frames for arbitrary MP4 videos.')
    parser.add_argument(
        '--videos', nargs='+', required=True,
        help='Video IDs with or without .mp4, for example test001')
    parser.add_argument('--data-root', default='data')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--every', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=1000)
    args = parser.parse_args()
    if args.every < 1 or args.chunk_size < 1:
        parser.error('--every and --chunk-size must be positive')
    return args


if __name__ == '__main__':
    arguments = parse_args()
    extract_videos(
        arguments.videos,
        data_root=arguments.data_root,
        overwrite=arguments.overwrite,
        every=arguments.every,
        chunk_size=arguments.chunk_size)
