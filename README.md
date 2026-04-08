# CoTracker → DeepLabCut exporter

This repository includes an interactive script that helps you:

1. Load an endoscope video.
2. Manually choose a frame range to track.
3. Manually click and name keypoints (e.g., matching points on identical DaVinci tools).
4. Run CoTracker on the selected range.
5. Export trajectories in a DeepLabCut-friendly format (`.csv` + `.h5`).

## Script

- `cotracker_dlc_tool.py`

## Install

```bash
pip install numpy pandas opencv-python torch
pip install git+https://github.com/facebookresearch/co-tracker.git
```

## Usage

```bash
python cotracker_dlc_tool.py \
  --video /path/to/endoscope_video.mp4 \
  --outdir outputs \
  --scorer surgeon1
```

### Interactive controls

#### 1) Frame range selector
- Use `Start` and `End` trackbars.
- `q` or `Enter`: confirm.
- `Esc`: abort.

#### 2) Keypoint selector
- Left click a keypoint location.
- Type a keypoint name in terminal prompt.
- `u`: undo last point.
- `q` or `Enter`: confirm (requires at least 1 keypoint).
- `Esc`: abort.

## Output files

For input `video.mp4`, outputs are:

- `outputs/video_dlc_labels.csv`
- `outputs/video_dlc_labels.h5`
- `outputs/video_tracking_metadata.json`

The CSV/H5 use a MultiIndex column structure compatible with DeepLabCut conventions:

- level 1: `scorer`
- level 2: `bodyparts`
- level 3: `coords` (`x`, `y`, `likelihood`)

Frame index names are written in DLC-like style:

- `labeled-data/<video_name>/img000123.png`

## Notes

- CoTracker outputs visibility/confidence; this is exported as DLC `likelihood`.
- If CUDA is unavailable or undesired, pass `--cpu`.
