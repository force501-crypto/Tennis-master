# Tennis MS-TCN 0010

This workspace uses a single MS-TCN pipeline for model `0010`.

```text
train.py                 train MS-TCN 0010
evaluate.py              evaluate only V006 and automatically export clips
prepare_features.py      extract RGB + flow features from frame model 0010
prepare_flow.py          regenerate missing optical-flow frames with FlowNet-S
dataset.py               load model 0010 feature sequences
models/vision/mstcn.py   MS-TCN++ temporal network
mstcn_utils.py           loss, metrics, and checkpoint helpers
export_class_clips.py    remove OTH and write one MP4 per class
analyze_predictions.py  optional prediction metrics and event analysis
process.py               optional video-to-frame preprocessing
```

The primary evaluation command is:

```bash
python evaluate.py --model_id 0010 --num_gpus 1
```

See `MSTCN.md` for training prerequisites and detailed output paths.
