#!/bin/bash
#SBATCH --job-name=topdown-cotracker
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Usage:
# sbatch run_topdown_cotracker.slurm /path/video.mp4 /path/annotations.json /path/output_dir

VIDEO_PATH="${1:?missing video path}"
ANNOT_JSON="${2:?missing annotation json path}"
OUTDIR="${3:-outputs}"
SCORER="${SCORER:-cotracker}"

mkdir -p "$(dirname "$OUTDIR")" logs

# Optional: activate env
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cotracker

python topdown_cotracker_dlc.py \
  --video "$VIDEO_PATH" \
  --annotation-json "$ANNOT_JSON" \
  --outdir "$OUTDIR" \
  --scorer "$SCORER"
