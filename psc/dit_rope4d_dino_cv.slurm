#!/bin/bash
#SBATCH -A bhgw-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --job-name=dit_rope4d_dino_cv
#SBATCH --output=psc/logs/%x_%j.out
#SBATCH --error=psc/logs/%x_%j.err

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate know_your_camera

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Ours — GROOT-DINO-CV-RoPE4D (DA3 weights + CameraEnc baked into the wrapper).
python policy_robosuite/train.py \
    --name dit_rope4d_dino_cv_square_seed_0 \
    --seed 0 \
    --policy_class dit_rope4d_dino_cv \
    --dataset_suffix squarerand_eef_delta \
    --num_side_cam 2 --n 4 --m 2 \
    --use_plucker 0 --prob_drop_proprio 0 \
    --num_epochs 30001 \
    --eval_start_epoch 0 \
    --eval_max_steps 600
