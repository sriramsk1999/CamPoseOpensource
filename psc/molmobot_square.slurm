#!/bin/bash

#SBATCH -N 1
#SBATCH -n 12
#SBATCH -p ROBO
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH -t 48:00:00
#SBATCH --job-name=molmobot_square
#SBATCH --output=psc/logs/%x_%j.out
#SBATCH --error=psc/logs/%x_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=sskrishn@andrew.cmu.edu

set -x

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate know_your_camera

cd "$SLURM_SUBMIT_DIR"

export ARTICUBOT_DP="$(cd "$SLURM_SUBMIT_DIR/../ArticuBot/diffusion_policy" && pwd)"

# Redirect HuggingFace cache to /ocean (project storage) — $HOME on PSC is
# ~25 GB and the MolmoBot-DROID checkpoint alone is ~20 GB. Without this,
# snapshot_download fails with "Not enough free disk space" mid-fetch.
export HF_HOME="${HF_HOME:-/ocean/projects/cis240052p/skrishne/.cache/huggingface}"
mkdir -p "$HF_HOME"

# MolmoBot baseline: Qwen3-4B MoE VLM + SigLIP2 + flow-matching action expert,
# vendored under ArticuBot/diffusion_policy/diffusion_policy/model/molmobot/.
# Vision encoder + projector frozen (paper recipe); LoRA on LLM attention +
# MLP; action expert trained full. Uses the joint_abs dataset (8-dim joint
# targets) built via script_robosuite_demos/build_joint_abs_dataset.py.
python policy_robosuite/train.py \
    --name molmobot_square_seed_0 \
    --seed 0 \
    --policy_class molmobot \
    --dataset_suffix squarerand_joint_abs \
    --num_side_cam 2 --n 4 --m 2 \
    --use_plucker 0 --prob_drop_proprio 0 \
    --batch_size 10 \
    --num_epochs 10001 \
    --eval_start_epoch 1000 \
    --eval_max_steps 600 \
    --lr 1e-5 --weight_decay 1e-6 --lr_scheduler cosine
