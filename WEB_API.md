# Tennis video Flask API

`web.py` accepts an MP4, runs the author's original model 0006 frame
classifier (`0015.params`) in a background worker, and returns downloadable
class-separated MP4 files. MS-TCN is not used.

## Start the server

```bash
source /home/thl/tennis-env/bin/activate
cd /mnt/f/Tennis-master
python -m pip install -r requirements-web.txt
python web.py
```

The default address is `http://0.0.0.0:5000`. Set `TENNIS_API_PORT` to change
the port. The default upload limit is 2 GiB and can be changed with
`TENNIS_MAX_UPLOAD_BYTES`.

## Upload

Send `multipart/form-data` with a file field named `file`:

```bash
curl -X POST -F "file=@test001.mp4" http://127.0.0.1:5000/upload
```

The API returns HTTP 202 with a `job_id`, `status_url`, and `results_url`.
Processing is asynchronous because frame extraction and GPU inference can take
many minutes.

## Poll status and fetch results

```bash
curl http://127.0.0.1:5000/jobs/JOB_ID
curl http://127.0.0.1:5000/jobs/JOB_ID/results
```

When `state` is `completed`, the results response contains one download URL
per detected class. Download a result using that URL:

```bash
curl -O http://127.0.0.1:5000/download/JOB_ID/SNF.mp4
```

For failures, inspect:

```bash
curl http://127.0.0.1:5000/jobs/JOB_ID/log
```

## Postman

1. Select POST and enter `http://SERVER_IP:5000/upload`.
2. Choose Body -> form-data.
3. Add key `file`, change its type from Text to File, and select an MP4.
4. Send the request and copy `job_id` from the JSON response.
5. Use GET on `/jobs/<job_id>` until `state` becomes `completed`.
6. Use GET on `/jobs/<job_id>/results`, then open a returned download URL.

Generated uploads, frames, predictions, logs, and videos remain outside Git.
