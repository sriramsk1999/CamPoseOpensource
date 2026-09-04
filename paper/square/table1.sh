#!/usr/bin/env bash
# One Table-1 arm, one seed, on one GPU. Runs in the FOREGROUND so run_all.sh
# can chain runs; nohup it yourself for a single job.
#
#   GPU=0 paper/square/table1.sh vgp 1
#   GPU=1 EPOCHS=201 paper/square/table1.sh plucker 1     # quick smoke test
#   DRY_RUN=1 paper/square/table1.sh vgp 1                # print the command
#
# Arms (Table 1, Square Assembly column). All three share the same cross-view
# DINOv2 encoder, the same DiT width/depth, and the same optimizer — the only
# thing that changes is how camera pose reaches the policy:
#
#   canonical  reproject the two chosen cameras into fixed canonical
#              viewpoints; the policy never sees a pose, the geometry is
#              baked into the RGB it is handed.
#   plucker    chosen-camera RGB as-is, 6-channel Plucker raymap fused into
#              the patch tokens (channel concat + project down).
#   vgp        chosen-camera RGB + metric pointmap; pose enters as RoPE4D
#              token positions, with action tokens anchored on the current
#              noisy trajectory estimate (action_pos_mode=cumsum).
#
# Resumes automatically: train.py picks up the newest epoch_*.pth in
# policy_robosuite/checkpoints/<name>/, so re-running the same arm+seed
# continues instead of starting over.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO_ROOT="$PWD"

ARM="${1:?usage: GPU=<n> paper/square/table1.sh <arm> <seed>  (see case block)}"
SEED="${2:?usage: GPU=<n> paper/square/table1.sh <arm> <seed>}"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-30001}"
BATCH_SIZE="${BATCH_SIZE:-35}"
# Only lower these for a smoke test; the paper numbers use the defaults.
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-600}"
DRY_RUN="${DRY_RUN:-0}"

# Same arms as paper/square/grogu.slurm -- keep the two in sync. Everything
# not named in an arm's block stays at the VGP configuration.
POLICY_CLASS=dit_rope4d_dino_cv; USE_PLUCKER=0; USE_CANON=0; EXTRA=()
case "$ARM" in
    # --- Table 1 -----------------------------------------------------------
    vgp)              ;;
    plucker)          POLICY_CLASS=dit_dino_cv; USE_PLUCKER=1 ;;
    canonical)        POLICY_CLASS=dit_dino_cv; USE_CANON=1 ;;
    # --- Table 2 -----------------------------------------------------------
    dp)               POLICY_CLASS=dp ;;
    act)              POLICY_CLASS=act ;;
    vgp_sinusoidal)   POLICY_CLASS=dit_dino_cv ;;
    spatial_softmax)  EXTRA=(--visual_token_compression softmax) ;;
    max_pool)         EXTRA=(--visual_token_compression max) ;;
    resnet18)         EXTRA=(--visual_encoder resnet18) ;;
    dinov2_frozen)    EXTRA=(--visual_encoder dinov2_frozen) ;;
    dinov2_finetuned) EXTRA=(--visual_encoder dinov2_finetuned) ;;
    rope3d)           EXTRA=(--pos_encoding rope3d) ;;
    *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

# train.py derives the wandb group as name[:-7], so the name MUST end in
# exactly "_seed_<N>" or runs land in the wrong group.
NAME="${ARM}_square_seed_${SEED}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate know_your_camera

# Cuts allocator fragmentation, which is what decides whether two of these fit
# on one card when JOBS_PER_GPU>1. Harmless at one job per GPU.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The dit_* wrappers import ArticuBot as a sidecar package off this path.
export ARTICUBOT_DP="${ARTICUBOT_DP:-$HOME/Desktop/ArticuBot/diffusion_policy}"
if [ ! -d "$ARTICUBOT_DP" ]; then
    echo "ARTICUBOT_DP not found: $ARTICUBOT_DP" >&2
    exit 1
fi

mkdir -p paper/square/logs
LOG="paper/square/logs/${NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "[$(date +%H:%M:%S)] GPU$GPU  $NAME  ($POLICY_CLASS, plucker=$USE_PLUCKER, canonical=$USE_CANON)  log=$LOG"

# Every flag that affects the run is passed explicitly rather than inherited
# from argparse defaults — several of those defaults do not match this recipe
# (lr 2e-5, lr_scheduler const, batch_size 70, prob_drop_proprio 1.0,
# num_side_cam 1, n 3, m 1).
CMD=(python "$REPO_ROOT/policy_robosuite/train.py"
    --name "$NAME"
    --seed "$SEED"
    --policy_class "$POLICY_CLASS"
    --dataset_suffix squarerand_eef_delta
    --use_plucker "$USE_PLUCKER"
    --use_canonical_views "$USE_CANON"
    --num_side_cam 2 --n 4 --m 2
    --default_cam 0 --use_cam_pose 0 --original 0
    --train_poses_file train_cameras.json
    --test_poses_file test_cameras.json
    --pose_files train_cameras.json test_cameras.json
    --num_episodes 200
    --prob_drop_proprio 0
    --horizon 16 --n_action_steps 8
    --batch_size "$BATCH_SIZE"
    --num_epochs "$EPOCHS"
    --lr 1e-4 --weight_decay 1e-6 --lr_scheduler cosine
    --use_fp16 1 --transform crop
    --save_every 5000
    --eval_every 1000 --eval_start_epoch 1000
    --eval_episodes "$EVAL_EPISODES" --eval_max_steps "$EVAL_MAX_STEPS"
    --eval_save_n_video 10
    "${EXTRA[@]}"
)

if [ "$DRY_RUN" != "0" ]; then
    echo "CUDA_VISIBLE_DEVICES=$GPU ${CMD[*]}"
    rm -f "$LOG"
    exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" >> "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] GPU$GPU  $NAME  done"
