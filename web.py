"""Flask API for uploading a tennis video and downloading classified clips.

The worker uses the author's original model 0006 frame classifier. MS-TCN is
not invoked. Long-running processing is asynchronous: POST /upload returns a
job id immediately and clients poll the job endpoints for results.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from flask import Flask, jsonify, request, send_from_directory, url_for
from werkzeug.utils import secure_filename


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / 'data'
VIDEO_ROOT = DATA_ROOT / 'videos'
JOB_ROOT = PROJECT_ROOT / 'api_jobs'
MODEL_PARAMS = (
    PROJECT_ROOT / 'models' / 'vision' / 'experiments' / '0006' /
    '0015.params')
ALLOWED_EXTENSIONS = {'.mp4'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(
    os.environ.get('TENNIS_MAX_UPLOAD_BYTES', 2 * 1024 * 1024 * 1024))

_executor = ThreadPoolExecutor(
    max_workers=int(os.environ.get('TENNIS_API_WORKERS', '1')))
_status_lock = threading.Lock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def job_dir(job_id):
    return JOB_ROOT / job_id


def status_path(job_id):
    return job_dir(job_id) / 'status.json'


def read_status(job_id):
    path = status_path(job_id)
    if not path.is_file():
        return None
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def write_status(job_id, **updates):
    """Atomically update a persistent job status file."""
    with _status_lock:
        current = read_status(job_id) or {'job_id': job_id}
        current.update(updates)
        current['updated_at'] = utc_now()
        directory = job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / 'status.json.tmp'
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(current, handle, ensure_ascii=False, indent=2)
        os.replace(str(temporary), str(status_path(job_id)))
    return current


def run_logged(command, log_handle, env=None):
    log_handle.write('$ {}\n'.format(' '.join(command)))
    log_handle.flush()
    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=True)


def process_job(job_id):
    directory = job_dir(job_id)
    output_dir = directory / 'outputs'
    prediction_file = directory / 'predictions_original.npz'
    log_path = directory / 'processing.log'
    python = sys.executable
    env = os.environ.copy()
    env.setdefault('MXNET_CUDNN_AUTOTUNE_DEFAULT', '0')
    env.setdefault('PYTHONUNBUFFERED', '1')

    try:
        write_status(job_id, state='extracting_frames', progress='抽取视频帧')
        with log_path.open('a', encoding='utf-8') as log_handle:
            run_logged([
                python, 'process.py', '--videos', job_id,
                '--data-root', str(DATA_ROOT),
            ], log_handle, env)

            write_status(
                job_id, state='running_model',
                progress='使用作者原始006模型进行逐帧分类')
            run_logged([
                python, 'evaluate_frame.py',
                '--video-id', job_id,
                '--model-id', '0006',
                '--num-gpus', os.environ.get('TENNIS_NUM_GPUS', '1'),
                '--num-workers', os.environ.get('TENNIS_NUM_WORKERS', '8'),
                '--batch-size', os.environ.get('TENNIS_BATCH_SIZE', '8'),
                '--params-file', str(MODEL_PARAMS),
                '--predictions-file', str(prediction_file),
                '--output-dir', str(output_dir),
                '--smooth-window', os.environ.get('TENNIS_SMOOTH_WINDOW', '9'),
                '--min-event-frames',
                os.environ.get('TENNIS_MIN_EVENT_FRAMES', '5'),
                '--min-event-confidence',
                os.environ.get('TENNIS_MIN_EVENT_CONFIDENCE', '0.50'),
                '--before-seconds',
                os.environ.get('TENNIS_BEFORE_SECONDS', '1'),
                '--after-seconds',
                os.environ.get('TENNIS_AFTER_SECONDS', '3'),
            ], log_handle, env)

        files = sorted(path.name for path in output_dir.glob('*.mp4'))
        if not files:
            raise RuntimeError('模型没有生成任何非 OTH 分类视频')
        write_status(
            job_id,
            state='completed',
            progress='处理完成',
            completed_at=utc_now(),
            result_files=files)
    except Exception as error:
        write_status(
            job_id,
            state='failed',
            progress='处理失败',
            error=str(error),
            log_file='processing.log')


def public_status(status):
    result = dict(status)
    job_id = result['job_id']
    result['status_url'] = url_for(
        'get_job', job_id=job_id, _external=True)
    result['results_url'] = url_for(
        'get_results', job_id=job_id, _external=True)
    return result


@app.get('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model': 'original-0006-frame-model',
        'checkpoint_exists': MODEL_PARAMS.is_file(),
    })


@app.post('/upload')
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'multipart/form-data 中缺少 file 字段'}), 400
    uploaded = request.files['file']
    original_name = secure_filename(uploaded.filename or '')
    suffix = Path(original_name).suffix.lower()
    if not original_name:
        return jsonify({'error': '文件名为空'}), 400
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({'error': '目前只支持 MP4 视频'}), 415
    if not MODEL_PARAMS.is_file():
        return jsonify({'error': '服务器缺少原始006模型参数'}), 503

    job_id = uuid.uuid4().hex
    directory = job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=False)
    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    video_path = VIDEO_ROOT / '{}.mp4'.format(job_id)
    uploaded.save(str(video_path))
    if video_path.stat().st_size == 0:
        video_path.unlink()
        return jsonify({'error': '上传的视频为空'}), 400

    status = write_status(
        job_id,
        state='queued',
        progress='等待处理',
        original_filename=original_name,
        stored_video='data/videos/{}.mp4'.format(job_id),
        created_at=utc_now())
    _executor.submit(process_job, job_id)
    return jsonify(public_status(status)), 202


@app.get('/jobs/<job_id>')
def get_job(job_id):
    status = read_status(job_id)
    if status is None:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify(public_status(status))


@app.get('/jobs/<job_id>/results')
def get_results(job_id):
    status = read_status(job_id)
    if status is None:
        return jsonify({'error': '任务不存在'}), 404
    if status.get('state') != 'completed':
        return jsonify(public_status(status)), 202
    files = [{
        'filename': filename,
        'download_url': url_for(
            'download_result', job_id=job_id, filename=filename,
            _external=True),
    } for filename in status.get('result_files', [])]
    return jsonify({'job_id': job_id, 'files': files})


@app.get('/download/<job_id>/<filename>')
def download_result(job_id, filename):
    status = read_status(job_id)
    if status is None:
        return jsonify({'error': '任务不存在'}), 404
    safe_name = secure_filename(filename)
    if safe_name != filename or not safe_name.lower().endswith('.mp4'):
        return jsonify({'error': '无效文件名'}), 400
    output_dir = job_dir(job_id) / 'outputs'
    if not (output_dir / safe_name).is_file():
        return jsonify({'error': '结果视频不存在'}), 404
    return send_from_directory(
        str(output_dir), safe_name, as_attachment=True,
        download_name=safe_name)


@app.get('/jobs/<job_id>/log')
def download_log(job_id):
    if read_status(job_id) is None:
        return jsonify({'error': '任务不存在'}), 404
    directory = job_dir(job_id)
    if not (directory / 'processing.log').is_file():
        return jsonify({'error': '处理日志尚未生成'}), 404
    return send_from_directory(
        str(directory), 'processing.log', as_attachment=False,
        mimetype='text/plain')


def init_app_storage():
    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    JOB_ROOT.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    init_app_storage()
    app.run(
        host=os.environ.get('TENNIS_API_HOST', '0.0.0.0'),
        port=int(os.environ.get('TENNIS_API_PORT', '5000')),
        debug=False,
        threaded=True)
