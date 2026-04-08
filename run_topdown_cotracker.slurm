#!/bin/bash
#SBATCH --job-name=topdown-cotracker-track
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Usage (headless GPU job):
# sbatch run_topdown_cotracker.slurm /cluster/path/video.mp4 /cluster/path/annotations.json /cluster/path/outdir

VIDEO_PATH="${1:?missing cluster video path}"
ANNOT_JSON="${2:?missing annotations json}"
OUTDIR="${3:?missing output dir}"
SCORER="${SCORER:-cotracker}"

mkdir -p logs "$OUTDIR"

python topdown_cotracker_dlc.py track \
  --annotations-json "$ANNOT_JSON" \
  --video "$VIDEO_PATH" \
  --outdir "$OUTDIR" \
  --scorer "$SCORER"
