# User Manual: Local GUI annotation → Cluster GPU tracking

This workflow is designed for your exact setup:

- **GUI runs only on your local machine** (laptop/workstation).
- You **copy annotation files + scripts to cluster**.
- Cluster runs **headless tracking only** (no GUI calls).
- You can now choose a **specific annotation frame** inside the selected frame range.

---
# install
```
git clone https://github.com/facebookresearch/co-tracker.git
cd co-tracker
pip install -e .

mkdir checkpoints
cd checkpoints
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
```

## 1) Local machine: do interactive annotation

### A. Top-down (2 tools: boxes first, then keypoints)
```bash
python topdown_cotracker_dlc.py annotate --video ./video_data/left2_resized.mp4 --annotations-out ./annotations/left2_topdown_annotations.json
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
scp topdown_cotracker_dlc.py run_topdown_cotracker.slurm ./annotations/left2_topdown_annotations.json eguan3@dsailogin.arch.jhu.edu:/home/eguan3/dlc-test/
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
  ./video_data/left2_resized.mp4 \
  ./left2_topdown_annotations.json \
  ./CoTrackerOutputs/ \
  Emily
```

Optional scorer:

```bash
SCORER=surgeon1 sbatch run_topdown_cotracker.slurm ...
```

---

## 4) Cluster: run keypoint-only tracking headlessly

```bash
python cotracker_dlc_tool.py track \
  --annotations-json ./left2_topdown_annotations.json \
  ./video_data/left2_resized.mp4 \
  --outdir ./CoTrackerOutputs/ \
  --scorer Emily
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
- `<video>_dlc_point_mapping.json` (maps DLC bodypart label -> tool + original point name)

### Keypoint-only outputs
- `<video>_dlc_labels.csv`
- `<video>_dlc_labels.h5`

Both DLC label outputs use MultiIndex columns (`scorer/bodyparts/coords`) and frame index style:
`labeled-data/<video_name>/imgXXXXXX.png`.

Top-down ownership behavior:
- Annotation JSON stores `tool` for every point.
- DLC export encodes ownership in bodypart labels as `tool_left__name` / `tool_right__name`.
- Duplicate names are allowed; duplicates are disambiguated with suffixes like `__dup2`.

---

## 7) DeepLabCut training quickstart

```python
import deeplabcut

config = "/path/to/config.yaml"
deeplabcut.create_training_dataset(config)
deeplabcut.train_network(config)
deeplabcut.evaluate_network(config)
```

---

## 8) Two-step top-down training in DLC (multi-animal, no YOLO)

Use a DLC multi-animal project where:
- `individuals = tool_left tool_right`
- `multianimalbodyparts = your keypoints` (same points for both identical tools)

New files:
- `train_two_step_dlc.py` (DLC-only helper)
- `run_two_step_dlc_train.slurm` (cluster training job)

### Step A (local, GUI): create project + labels

1) Create multi-animal project and generate `config.yaml`:

```bash
python train_two_step_dlc.py init-project  --project-name KBMT-test --experimenter Emily --videos ./video_data/left2_resized.mp4 --working-directory CoTrackerOutputs --individuals tool_left tool_right --bodyparts jaw1 jaw2 wrist1 wrist2 wrist3 shaft
```

2) Extract frames:

```bash
python train_two_step_dlc.py extract-frames --config /local/path/.../config.yaml
```

3) Label in DLC GUI:

```bash
python train_two_step_dlc.py label-gui --config /local/path/.../config.yaml
```

4) Optional label check:

```bash
python train_two_step_dlc.py check-labels --config /local/path/.../config.yaml
```

### Step B (cluster, headless GPU): train using existing config.yaml

Copy the entire DLC project folder (including labeled data + `config.yaml`) to cluster, then:

```bash
sbatch run_two_step_dlc_train.slurm /cluster/path/to/config.yaml
```

This SLURM job runs:
1) `create-trainset`
2) `train`
3) `evaluate`

### Inference (detector proposes boxes, pose predicts keypoints)

Run DLC inference on new videos:

```bash
python train_two_step_dlc.py analyze \
  --config /cluster/path/to/config.yaml \
  --videos /cluster/data/new_video.mp4 \
  --save-as-csv
```
