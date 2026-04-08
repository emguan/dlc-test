#!/bin/bash
#SBATCH --job-name=topdown-cotracker-track
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# --- Activate venv ---
source /home/eguan3/venvs/cotracker/bin/activate

# Optional debug
which python
python --version
python -c "import cv2; print('cv2 OK:', cv2.__version__)"

# --- Inputs ---
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
