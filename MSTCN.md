# MS-TCN 0010 pipeline

The project has one feature-preparation step, one training entry point, and one
evaluation entry point:

- `prepare_features.py` loads the existing DenseNet121 two-stream model 0010.
- `train.py` trains MS-TCN model `0010`.
- `evaluate.py` loads model `0010`, evaluates only `V006`, saves predictions,
  removes `OTH`, and exports every foreground class to a separate MP4.

Both entry points reject every model ID except `0010`.

## Required local assets

The existing frame model remains unchanged:

```text
models/vision/experiments/0010/0008.params
models/vision/experiments/0010/scores.txt
```

Because model 0010 was trained with `flow=twos`, optical-flow JPEG files must
exist under `data/flow/V006.mp4` through `V010.mp4`. If they are absent,
download the author-provided `FlowNet2-S_checkpoint.params`:

```bash
python -m pip install gdown
mkdir -p flownet
python -m gdown 1wA3lPxSPc4rKQoz6-8Pr077MulnHx0Sf \
  -O flownet/FlowNet2-S_checkpoint.params
```

Then generate the flow frames:

```bash
python prepare_flow.py \
  --videos V006,V007,V008,V009,V010 \
  --gpu_id 0
```

Existing flow JPEG files are skipped, so this command can be restarted safely.

`prepare_features.py` uses its RGB and optical-flow DenseNet backbones to write:

```text
data/features/0010/V006.mp4/<chunk>/<frame>.npy
```

The new temporal parameters are kept in a separate subdirectory so they cannot
overwrite `0008.params`:

```text
models/vision/experiments/0010/mstcn/*.params
models/vision/experiments/0010/mstcn/scores.txt
```

The program stops with a clear error if required RGB frames, optical flow,
`0010` features, or the MS-TCN checkpoint are missing.

## Prepare model 0010 features

Run this once before MS-TCN training:

```bash
python prepare_features.py \
  --model_id 0010 \
  --splits train,val,test_006_full \
  --num_gpus 1
```

Existing `.npy` features are skipped, so an interrupted extraction can safely
be restarted with the same command.

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
  0008.params
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
all-class video are generated.

For one MP4 per event instead of one MP4 per class:

```bash
python evaluate.py \
  --model_id 0010 \
  --num_gpus 1 \
  --clip_output_mode per_event
```

OpenCV preserves source FPS and resolution but does not copy audio.
