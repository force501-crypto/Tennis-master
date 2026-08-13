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
Events are processed chronologically. If the next event of the same class
starts inside that class's previous trailing three-second context, it is
skipped. Events with a different predicted class remain in the output so that
continuous rally actions are not lost.

Before clips are created, probabilities are smoothed over 9 frames. Events
shorter than 5 frames or with mean confidence below 0.50 are discarded. This
prevents a one-frame class flicker from becoming a four-second false clip.

`evaluate_frame.py` is an independent fallback that loads the author's
original model 0006 DenseNet frame classifier (`0015.params`) and performs
direct per-frame classification. It does not load or run MS-TCN.

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

## Run model 0006 on another video

Place `test001.mp4` under `data/videos/`, then run frame extraction, RGB
feature extraction, and MS-TCN inference with `--video_id test001`. Arbitrary
full videos do not require a split or annotation TXT file. Use
`--clips_output_dir` to choose the output folder, such as `test001_output`.
