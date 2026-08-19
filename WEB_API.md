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

The single-request `/upload` endpoint is intended for small files. For large
videos or an unstable public tunnel, use the resumable chunked uploader below.

## Resumable chunked upload (recommended for large videos)

`upload_chunked.ps1` uploads an MP4 in independent chunks, verifies every
chunk with SHA-256, retries failed chunks, and resumes by querying the chunks
already stored by the server. The default chunk size is 8 MiB.

Run this command from Windows PowerShell on the laptop:

```powershell
cd G:\MyProject\Tennis-master
powershell -ExecutionPolicy Bypass -File .\upload_chunked.ps1 `
  -FilePath "C:\Users\xiaoy\Desktop\test002.mp4" `
  -BaseUrl "http://tennisthl.sz1.natnps.cn"
```

If the command or network is interrupted, run the exact same command again.
The local resume record is stored under `%LOCALAPPDATA%\TennisUploader`, and
the server reports its received chunks before transmission resumes. Do not
change the file, URL, or chunk size between resume attempts.

When every chunk is present, the server assembles the original MP4, checks the
total byte count, deletes only the temporary chunk files, and returns the same
`job_id` used for status polling. Frame extraction and original model 0006
inference then start asynchronously.

The underlying HTTP API is:

```text
POST /uploads/init
GET  /uploads/<job_id>
PUT  /uploads/<job_id>/chunks/<chunk_index>
POST /uploads/<job_id>/complete
```

`POST /uploads/init` accepts JSON fields `original_filename`, `total_size`,
and `chunk_size`. Each chunk request uses an
`application/octet-stream` body and an optional `X-Chunk-Sha256` header.
The default server accepts chunks from 64 KiB through 16 MiB. The complete
request returns HTTP 409 with `missing_chunks` until every chunk is valid.

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
