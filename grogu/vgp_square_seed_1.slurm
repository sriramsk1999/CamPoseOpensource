#!/bin/bash
#SBATCH --job-name=vgp_square_seed_1
#SBATCH --partition=dheld
#SBATCH --constraint=rtx3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=grogu/logs/%x_%j.out
#SBATCH --error=grogu/logs/%x_%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#
# Table 1, Square Assembly, VGP arm, seed 1 -- one-off test submission on
# Grogu (dheld partition, rtx3090 to match the 24 GiB card the recipe was
# tuned against on the local 2x4090 box; see paper/square/run_all.sh).
#
# Every partition on Grogu is PreemptMode=REQUEUE; --requeue plus train.py's
# own "resume from newest checkpoint" behavior (paper/square/table1.sh) means
# a preempted job just picks back up.
#
#   sbatch grogu/vgp_square_seed_1.slurm
#   DRY_RUN=1 sbatch grogu/vgp_square_seed_1.slurm   # table1.sh prints the
#                                                     # command, launches nothing
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export ARTICUBOT_DP="$(cd "$SLURM_SUBMIT_DIR/../ArticuBot/diffusion_policy" && pwd)"

GPU=0 paper/square/table1.sh vgp 1
