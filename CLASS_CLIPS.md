# V006 class-separated video output

`evaluate.py` automatically performs the complete model `0010` workflow:

1. Load the best MS-TCN checkpoint from `models/vision/experiments/0010/`.
2. Read only `V006` feature sequences from `data/features/0010/`.
3. Save predictions to `predictions_V006.npz`.
4. Remove every `OTH` interval.
5. Write a separate chronological MP4 for each detected foreground class.

Run:

```bash
source /home/thl/tennis-env/bin/activate
cd /mnt/f/Tennis-master
python evaluate.py --model_id 0010 --num_gpus 1
```

Output is written to:

```text
models/vision/experiments/0010/class_clips/V006/
```

`manifest.csv` records the source frame interval, predicted class, confidence,
written frame count, and output filename for every event. Classes with no
detection are skipped; `OTH.mp4` is never created.

The standalone exporter remains available when a compatible prediction NPZ
already exists:

```bash
python export_class_clips.py \
  models/vision/experiments/0010/predictions_V006.npz \
  --video-id V006 \
  --output-dir models/vision/experiments/0010/class_clips/V006
```
