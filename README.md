# User Manual: Local GUI annotation → Cluster GPU tracking

This workflow is designed for your exact setup:

- **GUI runs only on your local machine** (laptop/workstation).
- You **copy annotation files + scripts to cluster**.
- Cluster runs **headless tracking only** (no GUI calls).
- You can now choose a **specific annotation frame** inside the selected frame range.

---

## 1) Local machine: do interactive annotation

### A. Top-down (2 tools: boxes first, then keypoints)
```bash
python topdown_cotracker_dlc.py annotate \
  --video /local/path/endoscope_video.mp4 \
  --annotations-out /local/path/topdown_annotations.json
```

Box selection behavior:
- Click+drag each ROI (release mouse to add ROI).
- Draw exactly 2 ROIs.
- Press `Enter` or `Space` to save.
- `u` undo last ROI, `r` reset all ROIs.

Annotation flow in order:
1. Select frame range.
2. Select the exact annotation frame within that range.
3. Draw ROIs and pick keypoints on that chosen frame.

### B. Keypoint-only (no boxes)
```bash
python cotracker_dlc_tool.py annotate \
  --video /local/path/endoscope_video.mp4 \
  --annotations-out /local/path/keypoints_annotations.json
```

This mode also prompts for frame range first, then lets you choose the exact annotation frame.

Outputs from this step are JSON files you can copy to cluster.

---

## 2) Copy required files to cluster

Example with `scp`:

```bash
scp topdown_cotracker_dlc.py run_topdown_cotracker.slurm /local/path/topdown_annotations.json user@cluster:/cluster/workdir/
```

Copy your video to cluster storage too (if not already present):

```bash
scp /local/path/endoscope_video.mp4 user@cluster:/cluster/data/
```

---

## 3) Cluster: submit headless GPU job (top-down)

SSH to cluster, then:

```bash
cd /cluster/workdir
sbatch run_topdown_cotracker.slurm \
  /cluster/data/endoscope_video.mp4 \
  /cluster/workdir/topdown_annotations.json \
  /cluster/workdir/outputs
```

Optional scorer:

```bash
SCORER=surgeon1 sbatch run_topdown_cotracker.slurm ...
```

---

## 4) Cluster: run keypoint-only tracking headlessly

```bash
python cotracker_dlc_tool.py track \
  --annotations-json /cluster/workdir/keypoints_annotations.json \
  --video /cluster/data/endoscope_video.mp4 \
  --outdir /cluster/workdir/outputs \
  --scorer surgeon1
```

---

## 5) Important path rule

The annotation JSON is created locally and can contain a local video path.
On cluster runs, always pass `--video /cluster/path/video.mp4` (or use the SLURM script which now requires cluster video path) so tracking uses the cluster file location.

---

## 6) Output files

### Top-down outputs
- `<video>_dlc_labels.csv`
- `<video>_dlc_labels.h5`
- `<video>_tool_boxes.csv`

### Keypoint-only outputs
- `<video>_dlc_labels.csv`
- `<video>_dlc_labels.h5`

Both DLC label outputs use MultiIndex columns (`scorer/bodyparts/coords`) and frame index style:
`labeled-data/<video_name>/imgXXXXXX.png`.

---

## 7) DeepLabCut training quickstart

```python
import deeplabcut

config = "/path/to/config.yaml"
deeplabcut.create_training_dataset(config)
deeplabcut.train_network(config)
deeplabcut.evaluate_network(config)
```
