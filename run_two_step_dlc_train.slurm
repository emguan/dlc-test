#!/bin/bash
#SBATCH --job-name=two-step-dlc-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Usage:
# sbatch run_two_step_dlc_train.slurm \
#   /cluster/data/video.mp4 \
#   /cluster/labels/video_tool_boxes.csv \
#   /cluster/labels/video_dlc_labels.csv \
#   /cluster/work/twostep \
#   /cluster/dlc/config.yaml

VIDEO_PATH="${1:?missing video path}"
TOOL_BOXES_CSV="${2:?missing tool boxes csv}"
DLC_CSV="${3:?missing dlc csv}"
WORKDIR="${4:?missing workdir}"
DLC_CONFIG="${5:?missing dlc config path}"

EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
DETECTOR_MODEL="${DETECTOR_MODEL:-yolov8n.pt}"

mkdir -p "$WORKDIR" logs

# 1) Prepare detector dataset + pose-crop dataset from existing DLC style outputs
python train_two_step_dlc.py prepare \
  --video "$VIDEO_PATH" \
  --tool-boxes-csv "$TOOL_BOXES_CSV" \
  --dlc-csv "$DLC_CSV" \
  --outdir "$WORKDIR"

# 2) Train detector (boxes)
python train_two_step_dlc.py train-detector \
  --data-yaml "$WORKDIR/detector_dataset/dataset.yaml" \
  --outdir "$WORKDIR/detector_runs" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --model "$DETECTOR_MODEL"

# 3) Train pose model (on crop labels prepared in WORKDIR/pose_crops_dataset)
# You should point DLC config to this crop dataset label source.
python train_two_step_dlc.py train-pose \
  --dlc-config "$DLC_CONFIG"
