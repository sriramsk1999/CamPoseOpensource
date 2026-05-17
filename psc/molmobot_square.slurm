#!/bin/bash
#SBATCH -A bhgw-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH --job-name=molmobot_square
#SBATCH --output=psc/logs/%x_%j.out
#SBATCH --error=psc/logs/%x_%j.err

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate know_your_camera

cd "$SLURM_SUBMIT_DIR"

export ARTICUBOT_DP="$(cd "$SLURM_SUBMIT_DIR/../ArticuBot/diffusion_policy" && pwd)"

# MolmoBot baseline: Qwen3-4B MoE VLM + SigLIP2 + flow-matching action expert,
# vendored under ArticuBot/diffusion_policy/diffusion_policy/model/molmobot/.
# Vision encoder + projector frozen (paper recipe); LoRA on LLM attention +
# MLP; action expert trained full. Uses the joint_abs dataset (8-dim joint
# targets) built via script_robosuite_demos/build_joint_abs_dataset.py.
#
# batch_size 2 + grad checkpointing fits a 40 GB A100. Bump to 4-8 on an
# 80 GB H100. num_epochs is intentionally low — replicating the paper's 200k
# steps across multiple GPUs at batch 1024 is not single-GPU-feasible; this
# is the pragmatic single-task fine-tune budget.
python policy_robosuite/train.py \
    --name molmobot_square_seed_0 \
    --seed 0 \
    --policy_class molmobot \
    --dataset_suffix squarerand_joint_abs \
    --num_side_cam 2 --n 4 --m 2 \
    --use_plucker 0 --prob_drop_proprio 0 \
    --batch_size 2 \
    --num_epochs 3001 \
    --eval_start_epoch 0 \
    --eval_max_steps 600 \
    --lr 1e-5 --weight_decay 1e-6 --lr_scheduler cosine
