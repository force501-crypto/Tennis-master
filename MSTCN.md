# MS-TCN 0006 RGB pipeline

The project has one feature-preparation step, one training entry point, and one
evaluation entry point:

- `prepare_features.py` loads the existing RGB DenseNet121 frame model `0006`
  and can scan every extracted V006 frame without using a split file.
- `train.py` trains MS-TCN with model `0006` RGB features.
- `evaluate.py` scans the complete extracted V006 video by default, saves
  predictions, removes `OTH`, and exports every detected foreground class to a
  separate MP4.

All three entry points use `0006` by default and reject other model IDs. No
optical-flow directory, FlowNet code, or FlowNet checkpoint is used.

## Required local assets

The existing frame model remains unchanged:

```text
models/vision/experiments/0006/0015.params
models/vision/experiments/0006/scores.txt
```

The normal RGB frames, split files, labels, and source V006 video must also be
present under `data/`:

```text
data/frames/V006.mp4/ ... data/frames/V010.mp4/
data/splits/02/
data/annotations/labels/
data/videos/V006.mp4
```

Feature preparation writes one pooled DenseNet feature per frame:

```text
data/features/0006/V006.mp4/<chunk>/<frame>.npy
```

Full-video feature preparation discovers frame numbers directly under
`data/frames/V006.mp4/`. It does not require those frames to appear in
`train.txt`, `val.txt`, `test.txt`, or `test_006_full.txt`.

The new temporal parameters are stored separately from `0015.params`:

```text
models/vision/experiments/0006/mstcn/*.params
models/vision/experiments/0006/mstcn/scores.txt
```

## 1. Prepare model 0006 RGB features

Activate the Ubuntu venv and enter the project:

```bash
source /home/thl/tennis-env/bin/activate
cd /mnt/f/Tennis-master
```

Extract the train, validation, and full V006 test features:

```bash
python prepare_features.py \
  --model_id 0006 \
  --splits train,val,test_006_full \
  --num_gpus 1
```

Existing `.npy` features are skipped. An interrupted extraction can therefore
be restarted with the same command. Use `--overwrite` only when deliberately
replacing all existing `0006` features.

To supplement the complete V006 video while skipping every existing feature:

```bash
python prepare_features.py \
  --model_id 0006 \
  --full_video_id V006 \
  --full_video_only \
  --num_gpus 1
```

## 2. Train MS-TCN

```bash
python train.py \
  --model_id 0006 \
  --num_gpus 1
```

Training uses the normal `train` and `val` splits. Its final check uses only
`V006` from `test_006_full`. Numeric checkpoints are written beneath
`models/vision/experiments/0006/mstcn/`.

## 3. Infer and cut the complete V006 video

```bash
python evaluate.py \
  --model_id 0006 \
  --num_gpus 1
```

The default is label-free full-video inference:

```text
full_video=true
video_id=V006
background_class=OTH
clip_output_mode=per_class
```

Because the complete video includes unlabeled frames and frames used during
training, this mode intentionally skips accuracy/F1 reporting. It is intended
for clip generation. To reproduce the labeled 25,549-frame test result, use:

```bash
python evaluate.py \
  --model_id 0006 \
  --nofull_video \
  --split test_006_full \
  --num_gpus 1
```

Evaluation produces:

```text
models/vision/experiments/0006/
  0015.params
  mstcn/
    predictions_V006_full.npz
    class_clips/V006_full/
      SFI.mp4
      SFF.mp4
      SFL.mp4
      SNI.mp4
      SNF.mp4
      SNL.mp4
      HFL.mp4
      HFR.mp4
      HNL.mp4
      HNR.mp4
      manifest.csv
```

Only classes detected in `V006` are created. No `OTH.mp4` and no combined
all-class video are generated. OpenCV preserves source FPS and resolution but
does not copy audio.

For one MP4 per event instead of one MP4 per class:

```bash
python evaluate.py \
  --model_id 0006 \
  --num_gpus 1 \
  --clip_output_mode per_event
```
