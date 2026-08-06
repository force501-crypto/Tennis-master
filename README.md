# Tennis MS-TCN 0006

This workspace uses one RGB-only MS-TCN pipeline based on frame model `0006`.
It does not require optical-flow images or FlowNet weights.

```text
prepare_features.py      extract RGB features with frame model 0006
train.py                 train MS-TCN with the extracted RGB features
evaluate.py              evaluate only V006 and automatically export clips
dataset.py               load RGB frames and model 0006 feature sequences
models/vision/mstcn.py   MS-TCN++ temporal network
mstcn_utils.py           loss, metrics, and checkpoint helpers
export_class_clips.py    remove OTH and write one MP4 per class
analyze_predictions.py  optional prediction metrics and event analysis
process.py               optional video-to-frame preprocessing
```

Run the pipeline in this order:

```bash
python prepare_features.py --model_id 0006 --num_gpus 1
python train.py --model_id 0006 --num_gpus 1
python evaluate.py --model_id 0006 --num_gpus 1
```

See `MSTCN.md` for prerequisites and detailed output paths.
