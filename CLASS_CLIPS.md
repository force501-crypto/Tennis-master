# V006 class-separated video output

`evaluate.py` performs the complete-video model `0006` RGB MS-TCN workflow:

1. Load the best MS-TCN checkpoint from
   `models/vision/experiments/0006/mstcn/`.
2. Discover every V006 frame directly under `data/frames/V006.mp4/`.
3. Read the corresponding features from `data/features/0006/`.
4. Save predictions to `predictions_V006_full.npz`.
5. Remove every `OTH` interval.
6. Write a separate chronological MP4 for each detected foreground class.

Each detected event is exported from `t-1s` through `t+3s`. The values are
converted to frames using the source video's FPS and clipped at video bounds.
Events are processed chronologically. If the next event starts inside the
previous exported clip's trailing three-second context, it is skipped so the
same source interval is not exported again, even when its predicted class is
different.

Run:

```bash
source /home/thl/tennis-env/bin/activate
cd /mnt/f/Tennis-master
python evaluate.py --model_id 0006 --num_gpus 1
```

Output is written to:

```text
models/vision/experiments/0006/mstcn/class_clips/V006_full/
```

`manifest.csv` records the source frame interval, predicted class, confidence,
written frame count, and output filename for every event. Classes with no
detection are skipped; `OTH.mp4` is never created.

The standalone exporter remains available when a compatible prediction NPZ
already exists:

```bash
python export_class_clips.py \
  models/vision/experiments/0006/mstcn/predictions_V006_full.npz \
  --video-id V006 \
  --before-seconds 1 \
  --after-seconds 3 \
  --output-dir models/vision/experiments/0006/mstcn/class_clips/V006_full
```
