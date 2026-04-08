# User Manual: Remote-GPU CoTracker with Local Interaction (Top-Down + Keypoint-Only)

This project is structured for your requirement:

- **Local interaction** (GUI): choose frame range, define boxes/keypoints, name points.
- **Remote GPU execution** (SLURM/headless): run CoTracker using saved annotation JSON.
- **DeepLabCut-ready exports**: CSV + H5 labels (and tool box tracks in top-down mode).

---

## 1) Which script should I use?

### A) Top-down workflow (recommended for your two DaVinci tools)
- Script: `topdown_cotracker_dlc.py`
- Strategy: **track tool boxes first (via box corners), then keypoints**.
- Output: `*_dlc_labels.csv`, `*_dlc_labels.h5`, and `*_tool_boxes.csv`.

### B) Keypoint-only workflow
- Script: `cotracker_dlc_tool.py`
- Strategy: track only manually selected keypoints.
- Output: `*_dlc_labels.csv`, `*_dlc_labels.h5`.

---

## 2) Install

```bash
pip install numpy pandas opencv-python torch
pip install git+https://github.com/facebookresearch/co-tracker.git
```

---

## 3) Top-down workflow (local annotate → remote GPU track)

## Step 3.1: Local interactive annotation
Run on a machine with display (your laptop/workstation):

```bash
python topdown_cotracker_dlc.py annotate --video ./video_data/left1_resized.mp4 --annotations-out ./annotations/left1_topdown_annotations.json
```

What you do interactively:
1. Select frame range.
2. Draw 2 boxes (`tool_left`, `tool_right`).
3. Click and name keypoints per tool.

This creates `topdown_annotations.json`.

## Step 3.2: Submit headless GPU tracking with SLURM

```bash
SCORER=emily sbatch run_topdown_cotracker.slurm ./annotations/left1_topdown_annotations.json /CoTrackerOutput/
```

Optional video override (if cluster path differs from local path):

```bash
SCORER=emily sbatch run_topdown_cotracker.slurm ./annotations/left1_topdown_annotations.json /CoTrackerOutput/
  ./video_data/left1_resized.mp4 
```

Optional scorer name:

```bash
SCORER=emily sbatch run_topdown_cotracker.slurm ...
```

---

## 4) Keypoint-only workflow (local annotate → remote GPU track)

## Step 4.1: Local interactive annotation

```bash
python cotracker_dlc_tool.py annotate \
  --video /path/endoscope_video.mp4 \
  --annotations-out /path/keypoints_annotations.json
```

## Step 4.2: Remote headless tracking

```bash
python cotracker_dlc_tool.py track \
  --annotations-json /path/keypoints_annotations.json \
  --video /cluster/path/endoscope_video.mp4 \
  --outdir /path/output_dir \
  --scorer surgeon1
```

---

## 5) Output reference

### Top-down outputs
- `<video>_dlc_labels.csv`
- `<video>_dlc_labels.h5`
- `<video>_tool_boxes.csv`

### Keypoint-only outputs
- `<video>_dlc_labels.csv`
- `<video>_dlc_labels.h5`

Both DLC label outputs follow MultiIndex columns:
- `scorer`
- `bodyparts`
- `coords` = `x`, `y`, `likelihood`

---

## 6) DeepLabCut training quickstart

1. Create/configure your DLC project.
2. Ensure labeled frame index paths align with DLC project structure:
   - `labeled-data/<video_name>/imgXXXXXX.png`
3. Use the generated `*_dlc_labels.csv` or `*_dlc_labels.h5` in your project.
4. Train/evaluate:

```python
import deeplabcut

config = "/path/to/config.yaml"
deeplabcut.create_training_dataset(config)
deeplabcut.train_network(config)
deeplabcut.evaluate_network(config)
```

---

## 7) Practical notes for remote clusters

- Annotation JSON is the contract between local GUI and remote GPU runs.
- If local/remote video paths differ, use `--video` override in `track` mode.
- SLURM script is **tracking-only** (no GUI), so it is safe for headless GPU nodes.
