#!/bin/bash

#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -o grogu_logs/job_%j.out
#SBATCH -e grogu_logs/job_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --mail-type=END

# Square Assembly runs on Grogu. One script for every arm; ARM picks one.
# Mirrors ArticuBot's paper/grogu/ablation.slurm, but this is the CamPose
# repo: conda not pixi, robosuite renders on the fly, and there is no dataset
# to stage (the demos hdf5 is 204 MB and lives in the repo).
#
#   sbatch --export=ALL,ARM=vgp,SEED=1 -p dheld -C rtx3090 -t 7-00:00:00 \
#          paper/square/grogu.slurm
#
# Resources are deliberately smaller than the ArticuBot script's -c 12/64G.
# CamPose runs num_workers=0 (the mujoco renderer is not fork-safe and the
# dataset holds a live env), so a job is ~1 busy core, and the dataset is a
# 204 MB hdf5 rather than a 68 GB tree. -c 4/24G lets all 8 GPUs on
# grogu-4-8 (281 GB) run at once; -c 12/64G would cap the node at 4 jobs.
#
# Normally submitted through paper/square/submit_grogu.sh.
#
# Preemption-safe the same way the ArticuBot jobs are: every Grogu partition is
# PreemptMode=REQUEUE, --requeue brings the job back, and train.py resumes from
# the newest epoch_*.pth in its ckpt_dir. SAVE_EVERY is 2000 here rather than
# the 5000 used locally -- an epoch is ~3.2 s, so 5000 would put ~4.4 h of work
# at risk on a preemptible queue.
set -x

ARM="${ARM:?set ARM -- see the case block below for the list}"
SEED="${SEED:?set SEED (1|2|3)}"

REPO="${CAMPOSE_REPO:-/project_data/held/$USER/CamPoseOpensource}"
export ARTICUBOT_DP="${ARTICUBOT_DP:-/project_data/held/mnakuraf/ArticuBot/diffusion_policy}"

cd "$REPO" || exit 1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate know_your_camera

# Headless rendering. The workstation these runs were developed on has an X
# server; a compute node does not, and mujoco silently produces black frames or
# dies without a GL platform. NOTHING in CamPose sets this -- if the arms train
# to a flat loss with 0% success, check here first.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# train.py calls wandb.init unconditionally and will hang on an interactive
# login prompt under a batch scheduler. Sync after the fact with `wandb sync`.
export WANDB_MODE="${WANDB_MODE:-offline}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EPOCHS="${EPOCHS:-30001}"
BATCH_SIZE="${BATCH_SIZE:-35}"
SAVE_EVERY="${SAVE_EVERY:-2000}"

# Arms. Table 1 is the first three, the rest are Table 2. Everything not named
# in an arm's block stays at the VGP configuration, so each row is one change.
POLICY=dit_rope4d_dino_cv; PLUCKER=0; CANON=0; EXTRA=()
case "$ARM" in
    # --- Table 1 -----------------------------------------------------------
    vgp)              ;;
    plucker)          POLICY=dit_dino_cv; PLUCKER=1 ;;
    canonical)        POLICY=dit_dino_cv; CANON=1 ;;
    # --- Table 2: compression ----------------------------------------------
    dp)               POLICY=dp ;;
    spatial_softmax)  EXTRA=(--visual_token_compression softmax) ;;
    max_pool)         EXTRA=(--visual_token_compression max) ;;
    # --- Table 2: visual encoder -------------------------------------------
    resnet18)         EXTRA=(--visual_encoder resnet18) ;;
    dinov2_frozen)    EXTRA=(--visual_encoder dinov2_frozen) ;;
    dinov2_finetuned) EXTRA=(--visual_encoder dinov2_finetuned) ;;
    # --- Table 2: positional encoding --------------------------------------
    act)              POLICY=act ;;
    # Additive sinusoidal PE. dit_dino_cv with neither plucker nor canonical
    # views is AdditivePosDiT + cross-view DINO on raw RGB -- which is what
    # ArticuBot's learned_sinusoidal arm actually trains, geometry and all.
    # It carries the confound documented in ArticuBot paper/icra/README.md:
    # dropping RoPE also removes the pointmap's only consumer, so this is not
    # a one-knob change from VGP. rope3d below is the clean comparison.
    vgp_sinusoidal)   POLICY=dit_dino_cv ;;
    rope3d)           EXTRA=(--pos_encoding rope3d) ;;
    *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

# name[:-7] becomes the wandb group, so it must end in exactly "_seed_<N>".
NAME="${ARM}_square_seed_${SEED}"

mkdir -p grogu_logs

python policy_robosuite/train.py \
    --name "$NAME" \
    --seed "$SEED" \
    --policy_class "$POLICY" \
    --dataset_suffix squarerand_eef_delta \
    --use_plucker "$PLUCKER" \
    --use_canonical_views "$CANON" \
    --num_side_cam 2 --n 4 --m 2 \
    --default_cam 0 --use_cam_pose 0 --original 0 \
    --train_poses_file train_cameras.json \
    --test_poses_file test_cameras.json \
    --pose_files train_cameras.json test_cameras.json \
    --num_episodes 200 \
    --prob_drop_proprio 0 \
    --horizon 16 --n_action_steps 8 \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$EPOCHS" \
    --lr 1e-4 --weight_decay 1e-6 --lr_scheduler cosine \
    --use_fp16 1 --transform crop \
    --save_every "$SAVE_EVERY" \
    --eval_every 1000 --eval_start_epoch 1000 \
    --eval_episodes 10 --eval_max_steps 600 --eval_save_n_video 10 \
    "${EXTRA[@]}"
