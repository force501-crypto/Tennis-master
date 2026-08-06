# MS-TCN 0010 pipeline

The project now has one training entry point and one evaluation entry point:

- `train.py` trains MS-TCN model `0010`.
- `evaluate.py` loads model `0010`, evaluates only `V006`, saves predictions,
  removes `OTH`, and exports every foreground class to a separate MP4.

Both entry points reject every model ID except `0010`.

## Required local assets

The temporal network consumes pre-extracted feature arrays belonging to model
`0010`:

```text
data/features/0010/V006.mp4/<chunk>/<frame>.npy
```

Its trained parameters are read from:

```text
models/vision/experiments/0010/*.params
models/vision/experiments/0010/scores.txt
```

The program stops with a clear error if either the `0010` features or the
`0010` MS-TCN checkpoint are missing. It never falls back to another model.

## Train model 0010

Activate the Ubuntu venv and enter the project:

```bash
source /home/thl/tennis-env/bin/activate
cd /mnt/f/Tennis-master
```

Train:

```bash
python train.py \
  --model_id 0010 \
  --num_gpus 1
```

Training uses the normal `train` and `val` splits. Its final check uses only
`V006` from `test_006_full`.

## Evaluate and cut V006

```bash
python evaluate.py \
  --model_id 0010 \
  --num_gpus 1
```

Defaults are deliberately restricted to:

```text
split=test_006_full
video_id=V006
background_class=OTH
clip_output_mode=per_class
```

Evaluation produces:

```text
models/vision/experiments/0010/
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
all-class video are generated.

For one MP4 per event instead of one MP4 per class:

```bash
python evaluate.py \
  --model_id 0010 \
  --num_gpus 1 \
  --clip_output_mode per_event
```

OpenCV preserves source FPS and resolution but does not copy audio.
