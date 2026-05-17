#!/bin/bash
#SBATCH -A bhgw-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --job-name=act_dino_sv
#SBATCH --output=psc/logs/%x_%j.out
#SBATCH --error=psc/logs/%x_%j.err

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate know_your_camera

# Slurm copies the script to a temp scratch dir, so $0 doesn't point at the
# repo. $SLURM_SUBMIT_DIR is the directory you ran sbatch from — submit_all.sh
# cd's to the repo root before each sbatch call, so this lands at the repo.
cd "$SLURM_SUBMIT_DIR"

# ArticuBot is checked out as a sibling of CamPoseOpensource on the cluster
# (defaults to ~/Desktop/ArticuBot on the dev box). Point sidecar at it.
export ARTICUBOT_DP="$(cd "$SLURM_SUBMIT_DIR/../ArticuBot/diffusion_policy" && pwd)"

python policy_robosuite/train.py \
    --name act_dino_sv_plucker_square_seed_0 \
    --seed 0 \
    --policy_class act_dino_sv \
    --dataset_suffix squarerand_eef_delta \
    --num_side_cam 2 --n 4 --m 2 \
    --use_plucker 1 --prob_drop_proprio 0 \
    --num_epochs 30001 \
    --eval_start_epoch 1000 \
    --eval_max_steps 600
