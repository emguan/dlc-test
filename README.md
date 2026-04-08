# Top-down CoTracker → DeepLabCut pipeline

This repo now supports a **top-down workflow** for your 2 identical DaVinci tools:

1. Pick a frame range to track.
2. Define one bounding box per tool (left/right).
3. Define named keypoints per tool.
4. Track both tool boxes and keypoints with CoTracker.
5. Export DeepLabCut labels for training.

## Files

- `topdown_cotracker_dlc.py` (main pipeline)
- `run_topdown_cotracker.slurm` (SLURM submit script)
- `cotracker_dlc_tool.py` (previous simpler interactive script)

## Install

```bash
pip install numpy pandas opencv-python torch
pip install git+https://github.com/facebookresearch/co-tracker.git
```

## Step A: Build / save annotations (interactive)

Run this on a machine with display access:

```bash
python topdown_cotracker_dlc.py \
  --video /path/endoscope.mp4 \
  --save-annotation-json /path/annotations.json \
  --outdir /tmp/dry_run_outputs
```

This writes `annotations.json` containing:
- `frame_start`, `frame_end`
- `boxes` (`tool_left`, `tool_right`)
- named keypoints (`name`, `x`, `y`, `tool`)

## Step B: Run as SLURM job (non-interactive)

```bash
sbatch run_topdown_cotracker.slurm \
  /path/endoscope.mp4 \
  /path/annotations.json \
  /path/output_dir
```

Environment variable override:

```bash
SCORER=surgeon1 sbatch run_topdown_cotracker.slurm ...
```

## Outputs

For `video.mp4`, generated files include:

- `video_dlc_labels.csv` (DLC MultiIndex labels)
- `video_dlc_labels.h5` (DLC `df_with_missing` table)
- `video_tool_boxes.csv` (top-down tracked bounding boxes per frame/tool)

## Training DeepLabCut from this output

1. Create a DLC project.
2. Put your video frames under DLC `labeled-data/<video_name>/` if needed.
3. Place the generated `*_dlc_labels.csv` (or `.h5`) into the project label structure.
4. Ensure bodypart names match your intended keypoint schema.
5. Run the normal DLC pipeline:

```python
import deeplabcut

config = "/path/to/project/config.yaml"

deeplabcut.create_training_dataset(config)
deeplabcut.train_network(config)
deeplabcut.evaluate_network(config)
```

## Notes

- The top-down design is implemented by tracking **box corners first**, then keypoints.
- Tool box trajectories are exported separately to `*_tool_boxes.csv`.
- For SLURM jobs, use `--annotation-json` to avoid any GUI interaction.
