"""Flask API for uploading a tennis video and downloading classified clips.

The worker uses the author's original model 0006 frame classifier. MS-TCN is
not invoked. Long-running model processing is asynchronous. Small files can
use POST /upload; large files can use resumable chunk endpoints before clients
poll the job endpoints for results.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
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
DEFAULT_CHUNK_BYTES = int(
    os.environ.get('TENNIS_UPLOAD_CHUNK_BYTES', 8 * 1024 * 1024))
MAX_CHUNK_BYTES = int(
    os.environ.get('TENNIS_MAX_CHUNK_BYTES', 16 * 1024 * 1024))
MIN_CHUNK_BYTES = 64 * 1024

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(
    os.environ.get('TENNIS_MAX_UPLOAD_BYTES', 2 * 1024 * 1024 * 1024))

_executor = ThreadPoolExecutor(
    max_workers=int(os.environ.get('TENNIS_API_WORKERS', '1')))
_status_lock = threading.Lock()
_upload_lock = threading.Lock()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def job_dir(job_id):
    return JOB_ROOT / job_id


def status_path(job_id):
    return job_dir(job_id) / 'status.json'


def chunks_dir(job_id):
    return job_dir(job_id) / 'chunks'


def chunk_path(job_id, chunk_index):
    return chunks_dir(job_id) / '{:08d}.part'.format(chunk_index)


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


def positive_int(value, field_name):
    if isinstance(value, bool):
        raise ValueError('{} 必须是正整数'.format(field_name))
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError('{} 必须是正整数'.format(field_name))
    if result <= 0:
        raise ValueError('{} 必须是正整数'.format(field_name))
    return result


def expected_chunk_bytes(status, chunk_index):
    chunk_count = int(status['chunk_count'])
    if chunk_index < 0 or chunk_index >= chunk_count:
        raise IndexError('分片编号超出范围')
    chunk_size = int(status['chunk_size'])
    total_size = int(status['total_bytes'])
    offset = chunk_index * chunk_size
    return min(chunk_size, total_size - offset)


def chunk_inventory(status):
    """Return valid received chunks and their total bytes."""
    job_id = status['job_id']
    received = []
    uploaded_bytes = 0
    for chunk_index in range(int(status['chunk_count'])):
        path = chunk_path(job_id, chunk_index)
        if not path.is_file():
            continue
        expected = expected_chunk_bytes(status, chunk_index)
        if path.stat().st_size != expected:
            continue
        received.append(chunk_index)
        uploaded_bytes += expected
    return received, uploaded_bytes


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


def public_upload_status(status):
    result = public_status(status)
    received, uploaded_bytes = chunk_inventory(status)
    total_bytes = int(status['total_bytes'])
    if (status.get('state') not in ('uploading', 'assembling') and
            int(status.get('uploaded_bytes') or 0) == total_bytes):
        received = list(range(int(status['chunk_count'])))
        uploaded_bytes = total_bytes
    result.update({
        'upload_status_url': url_for(
            'get_chunked_upload', job_id=status['job_id'], _external=True),
        'complete_url': url_for(
            'complete_chunked_upload', job_id=status['job_id'],
            _external=True),
        'chunk_url_template': url_for(
            'put_upload_chunk', job_id=status['job_id'], chunk_index=0,
            _external=True).rsplit('/0', 1)[0] + '/{chunk_index}',
        'received_chunks': received,
        'received_chunk_count': len(received),
        'uploaded_bytes': uploaded_bytes,
        'upload_percent': round(100.0 * uploaded_bytes / total_bytes, 2),
    })
    return result


def queue_job(job_id, original_name, created_at=None):
    current = read_status(job_id) or {}
    updates = {
        'state': 'queued',
        'progress': '等待处理',
        'original_filename': original_name,
        'stored_video': 'data/videos/{}.mp4'.format(job_id),
        'created_at': created_at or utc_now(),
        'error': None,
    }
    if current.get('total_bytes') is not None:
        updates['uploaded_bytes'] = current['total_bytes']
    status = write_status(job_id, **updates)
    _executor.submit(process_job, job_id)
    return status


@app.get('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model': 'original-0006-frame-model',
        'checkpoint_exists': MODEL_PARAMS.is_file(),
        'chunked_upload': True,
        'recommended_chunk_bytes': DEFAULT_CHUNK_BYTES,
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

    status = queue_job(job_id, original_name)
    return jsonify(public_status(status)), 202


@app.post('/uploads/init')
def init_chunked_upload():
    if not MODEL_PARAMS.is_file():
        return jsonify({'error': '服务器缺少原始006模型参数'}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': '请求体必须是 JSON'}), 400

    original_name = secure_filename(str(payload.get('original_filename', '')))
    if not original_name:
        return jsonify({'error': '文件名为空'}), 400
    if Path(original_name).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify({'error': '目前只支持 MP4 视频'}), 415

    try:
        total_size = positive_int(payload.get('total_size'), 'total_size')
        chunk_size = positive_int(
            payload.get('chunk_size', DEFAULT_CHUNK_BYTES), 'chunk_size')
    except ValueError as error:
        return jsonify({'error': str(error)}), 400
    if total_size > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({
            'error': '视频超过服务器允许的最大大小',
            'max_bytes': app.config['MAX_CONTENT_LENGTH'],
        }), 413
    if chunk_size < MIN_CHUNK_BYTES or chunk_size > MAX_CHUNK_BYTES:
        return jsonify({
            'error': 'chunk_size 必须在 {} 到 {} 字节之间'.format(
                MIN_CHUNK_BYTES, MAX_CHUNK_BYTES),
        }), 400

    chunk_count = (total_size + chunk_size - 1) // chunk_size
    job_id = uuid.uuid4().hex
    directory = job_dir(job_id)
    directory.mkdir(parents=True, exist_ok=False)
    chunks_dir(job_id).mkdir()
    status = write_status(
        job_id,
        state='uploading',
        progress='等待分片上传',
        original_filename=original_name,
        total_bytes=total_size,
        uploaded_bytes=0,
        chunk_size=chunk_size,
        chunk_count=chunk_count,
        created_at=utc_now())
    return jsonify(public_upload_status(status)), 201


@app.get('/uploads/<job_id>')
def get_chunked_upload(job_id):
    status = read_status(job_id)
    if status is None or 'chunk_count' not in status:
        return jsonify({'error': '分片上传任务不存在'}), 404
    return jsonify(public_upload_status(status))


@app.put('/uploads/<job_id>/chunks/<int:chunk_index>')
def put_upload_chunk(job_id, chunk_index):
    status = read_status(job_id)
    if status is None or 'chunk_count' not in status:
        return jsonify({'error': '分片上传任务不存在'}), 404
    if status.get('state') != 'uploading':
        return jsonify({
            'error': '任务当前不接受分片',
            'state': status.get('state'),
        }), 409
    try:
        expected_size = expected_chunk_bytes(status, chunk_index)
    except IndexError as error:
        return jsonify({'error': str(error)}), 416
    if request.content_length is not None and request.content_length != expected_size:
        return jsonify({
            'error': '分片大小错误',
            'expected_bytes': expected_size,
            'received_bytes': request.content_length,
        }), 400

    directory = chunks_dir(job_id)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / '{}.tmp'.format(uuid.uuid4().hex)
    digest = hashlib.sha256()
    received_size = 0
    try:
        with temporary.open('wb') as handle:
            while True:
                block = request.stream.read(1024 * 1024)
                if not block:
                    break
                received_size += len(block)
                if received_size > expected_size:
                    raise ValueError('分片数据超过预期大小')
                handle.write(block)
                digest.update(block)
        if received_size != expected_size:
            raise ValueError(
                '分片不完整：预期 {} 字节，收到 {} 字节'.format(
                    expected_size, received_size))
        expected_digest = request.headers.get('X-Chunk-Sha256', '').lower()
        actual_digest = digest.hexdigest()
        if expected_digest and expected_digest != actual_digest:
            raise ValueError('分片 SHA-256 校验失败')

        with _upload_lock:
            os.replace(str(temporary), str(chunk_path(job_id, chunk_index)))
            received, uploaded_bytes = chunk_inventory(status)
            status = write_status(
                job_id,
                progress='已上传 {}/{} 个分片'.format(
                    len(received), status['chunk_count']),
                uploaded_bytes=uploaded_bytes)
    except (OSError, ValueError) as error:
        if temporary.exists():
            temporary.unlink()
        return jsonify({'error': str(error)}), 400

    result = public_upload_status(status)
    result['accepted_chunk'] = chunk_index
    result['chunk_sha256'] = actual_digest
    return jsonify(result)


@app.post('/uploads/<job_id>/complete')
def complete_chunked_upload(job_id):
    status = read_status(job_id)
    if status is None or 'chunk_count' not in status:
        return jsonify({'error': '分片上传任务不存在'}), 404
    if status.get('state') not in ('uploading', 'assembling'):
        return jsonify(public_status(status)), 200

    with _upload_lock:
        status = read_status(job_id)
        received, uploaded_bytes = chunk_inventory(status)
        chunk_count = int(status['chunk_count'])
        if len(received) != chunk_count:
            missing = sorted(set(range(chunk_count)) - set(received))
            return jsonify({
                'error': '仍有分片未上传',
                'received_chunk_count': len(received),
                'chunk_count': chunk_count,
                'missing_chunks': missing,
            }), 409

        write_status(
            job_id,
            state='assembling',
            progress='合并上传分片',
            uploaded_bytes=uploaded_bytes)
        VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
        video_path = VIDEO_ROOT / '{}.mp4'.format(job_id)
        temporary_video = VIDEO_ROOT / '.{}.assembling'.format(job_id)
        try:
            with temporary_video.open('wb') as output:
                for chunk_index in range(chunk_count):
                    with chunk_path(job_id, chunk_index).open('rb') as source:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
            if temporary_video.stat().st_size != int(status['total_bytes']):
                raise RuntimeError('合并后的视频大小与原文件不一致')
            os.replace(str(temporary_video), str(video_path))
        except Exception as error:
            if temporary_video.exists():
                temporary_video.unlink()
            write_status(
                job_id, state='uploading', progress='分片合并失败',
                error=str(error))
            return jsonify({'error': str(error)}), 500

        shutil.rmtree(str(chunks_dir(job_id)), ignore_errors=True)
        status = queue_job(
            job_id, status['original_filename'], status.get('created_at'))
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
