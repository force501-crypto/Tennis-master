"""Generate optical-flow frames required by the model 0010 two-stream CNN."""
import os

os.environ.setdefault('MXNET_CUDNN_AUTOTUNE_DEFAULT', '0')

from absl import app, flags
from absl.flags import FLAGS
import mxnet as mx

from models.vision.flownet.model import get_flownet
from models.vision.flownet.run import process_imagedir


flags.DEFINE_list(
    'videos', 'V006,V007,V008,V009,V010',
    'Videos whose optical-flow frames are generated.')
flags.DEFINE_string('data_root', 'data', 'Dataset root.')
flags.DEFINE_string(
    'params_file', 'flownet/FlowNet2-S_checkpoint.params',
    'Converted FlowNet2-S checkpoint from the project author.')
flags.DEFINE_integer('gpu_id', 0, 'GPU index; use -1 for CPU.')
flags.DEFINE_bool('overwrite', False, 'Replace existing flow JPEG files.')


def main(_argv):
    if not FLAGS.videos:
        raise ValueError('At least one video is required')
    if not os.path.exists(FLAGS.params_file):
        raise FileNotFoundError(
            'FlowNet checkpoint is missing: {}'.format(FLAGS.params_file))

    context = mx.gpu(FLAGS.gpu_id) if FLAGS.gpu_id >= 0 else mx.cpu()
    checkpoint_dir = os.path.dirname(os.path.abspath(FLAGS.params_file))
    checkpoint_name = os.path.basename(FLAGS.params_file)
    if checkpoint_name != 'FlowNet2-S_checkpoint.params':
        raise ValueError(
            'Expected FlowNet2-S_checkpoint.params, got {}'.format(
                checkpoint_name))
    model = get_flownet(ctx=context, root=checkpoint_dir)
    model.hybridize()
    print('Loaded FlowNet parameters: {}'.format(
        os.path.abspath(FLAGS.params_file)))
    print('Context: {}'.format(context))

    for video in FLAGS.videos:
        input_dir = os.path.join(
            FLAGS.data_root, 'frames', '{}.mp4'.format(video))
        output_dir = os.path.join(
            FLAGS.data_root, 'flow', '{}.mp4'.format(video))
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(
                'Extracted RGB frames are missing: {}'.format(input_dir))
        print('Generating {} flow: {} -> {}'.format(
            video, input_dir, output_dir))
        process_imagedir(
            model, input_dir=input_dir, output_dir=output_dir,
            ctx=context, overwrite=FLAGS.overwrite)
    print('Optical-flow generation complete.')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
