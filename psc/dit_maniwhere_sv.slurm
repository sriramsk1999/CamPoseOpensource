#!/bin/bash
#SBATCH -A bhgw-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --job-name=dit_maniwhere_sv
#SBATCH --output=psc/logs/%x_%j.out
#SBATCH --error=psc/logs/%x_%j.err

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate know_your_camera

cd "$SLURM_SUBMIT_DIR"

export ARTICUBOT_DP="$(cd "$SLURM_SUBMIT_DIR/../ArticuBot/diffusion_policy" && pwd)"

python policy_robosuite/train.py \
    --name dit_maniwhere_sv_square_seed_0 \
    --seed 0 \
    --policy_class dit_maniwhere_sv \
    --dataset_suffix squarerand_eef_delta \
    --num_side_cam 2 --n 4 --m 2 \
    --use_plucker 0 --prob_drop_proprio 0 \
    --num_epochs 30001 \
    --eval_start_epoch 0 \
    --eval_max_steps 600
