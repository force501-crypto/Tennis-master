# Tennis MS-TCN 0006

This workspace uses one RGB-only MS-TCN pipeline based on frame model `0006`.
It does not require optical-flow images or FlowNet weights.

```text
prepare_features.py      extract split features or every V006 RGB frame
train.py                 train MS-TCN with the extracted RGB features
evaluate.py              scan full V006 and automatically export clips
dataset.py               load RGB frames and model 0006 feature sequences
models/vision/mstcn.py   MS-TCN++ temporal network
mstcn_utils.py           loss, metrics, and checkpoint helpers
export_class_clips.py    remove OTH and write one MP4 per class
analyze_predictions.py  optional prediction metrics and event analysis
process.py               optional video-to-frame preprocessing
```

For full-video V006 inference, supplement any missing frame features and then
run the trained MS-TCN checkpoint:

```bash
python prepare_features.py --model_id 0006 --full_video_id V006 --full_video_only --num_gpus 1
python evaluate.py --model_id 0006 --num_gpus 1
```

See `MSTCN.md` for prerequisites and detailed output paths.
