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
# sbatch run_topdown_cotracker.slurm /path/annotations.json /path/outdir [/path/video_override.mp4]

ANNOT_JSON="${1:?missing annotations json}"
OUTDIR="${2:?missing output dir}"
VIDEO_OVERRIDE="${3:-}"
SCORER="${SCORER:-cotracker}"

mkdir -p logs "$OUTDIR"

CMD=(python topdown_cotracker_dlc.py track --annotations-json "$ANNOT_JSON" --outdir "$OUTDIR" --scorer "$SCORER")
if [[ -n "$VIDEO_OVERRIDE" ]]; then
  CMD+=(--video "$VIDEO_OVERRIDE")
fi

"${CMD[@]}"
