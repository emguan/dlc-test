#!/bin/bash
#SBATCH --job-name=dlc-multianimal-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# IMPORTANT: This script requires an existing DLC config.yaml.
# Create it first (typically local machine with GUI access) using:
#   python train_two_step_dlc.py init-project ...
# Then label frames in DLC GUI and sync project folder to cluster.
#
# Usage:
# sbatch run_two_step_dlc_train.slurm /cluster/path/to/config.yaml

DLC_CONFIG="${1:?missing DLC config.yaml path}"

mkdir -p logs

python train_two_step_dlc.py create-trainset --config "$DLC_CONFIG"
python train_two_step_dlc.py train --config "$DLC_CONFIG"
python train_two_step_dlc.py evaluate --config "$DLC_CONFIG"
