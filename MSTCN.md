# MS-TCN 0006 RGB pipeline

The project has one feature-preparation step, one training entry point, and one
evaluation entry point:

- `prepare_features.py` loads the existing RGB DenseNet121 frame model `0006`.
- `train.py` trains MS-TCN with model `0006` RGB features.
- `evaluate.py` evaluates only `V006`, saves predictions, removes `OTH`, and
  exports every detected foreground class to a separate MP4.

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

## 2. Train MS-TCN

```bash
python train.py \
  --model_id 0006 \
  --num_gpus 1
```

Training uses the normal `train` and `val` splits. Its final check uses only
`V006` from `test_006_full`. Numeric checkpoints are written beneath
`models/vision/experiments/0006/mstcn/`.

## 3. Evaluate and cut V006

```bash
python evaluate.py \
  --model_id 0006 \
  --num_gpus 1
```

Defaults are restricted to:

```text
split=test_006_full
video_id=V006
background_class=OTH
clip_output_mode=per_class
```

Evaluation produces:

```text
models/vision/experiments/0006/
  0015.params
  mstcn/
    predictions_V006.npz
    class_clips/V006/
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
