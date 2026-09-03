#!/usr/bin/env bash
# Table 1, Square Assembly: 3 arms x 3 seeds = 9 runs across the 2x4090 box.
#
#   paper/square/run_all.sh                  # all 9, in background
#   JOBS_PER_GPU=2 paper/square/run_all.sh   # pack 2 concurrent runs per GPU
#   DRY_RUN=1 paper/square/run_all.sh        # print the plan, launch nothing
#   SEEDS=1 paper/square/run_all.sh          # one seed (3 runs)
#   ARMS="vgp plucker" paper/square/run_all.sh
#
# Jobs are dealt round-robin across NGPU * JOBS_PER_GPU lanes; each lane runs
# its jobs one after another on its assigned GPU. The queue is ordered
# seed-major, so seed 1 finishes for all three arms before seed 2 starts — a
# partial sweep still gives you a complete table row.
#
# ~40 h per run at 1 job/GPU (measured: 30k epochs, dit_rope4d_dino_cv, this
# box), so the full sweep is roughly 8 days serialised 5/4 across the two GPUs.
# Interrupting is safe: re-running this script resumes each job from its newest
# checkpoint.
#
# JOBS_PER_GPU > 1 trades memory headroom for throughput. Tempting, because the
# GPU is badly starved at 1 job/GPU — measured mean utilisation 40%, 79% of
# samples under 50%, since num_workers=0 makes mujoco render every frame on the
# main process. But it was tried at batch 35 and does NOT fit: steady-state
# training is 14.5 GiB (vgp) / 16.5 GiB (plucker) per job on a 24 GiB card.
# The failure is delayed and easy to misread — a job sits at ~3 GiB through its
# epoch-0 eval (no_grad, no optimizer state) and only OOMs when it enters the
# training loop, so a pair can look fine for 20 minutes before one dies. See
# paper/square/README.md. Needs batch ~16-18 to be viable.
#
# Disk: ~12 GB of checkpoints per run (4 GB x keep-3), ~110 GB for all nine.
set -euo pipefail
cd "$(dirname "$0")/../.."

SEEDS="${SEEDS:-1 2 3}"
ARMS="${ARMS:-vgp plucker canonical}"
GPUS="${GPUS:-0 1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
DRY_RUN="${DRY_RUN:-0}"
# Seconds to stagger lane startups, so two jobs sharing a GPU do not hit their
# allocation peaks at the same instant.
STAGGER="${STAGGER:-90}"

read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

# One lane = one sequential worker. Lane i runs on GPU[i % NGPU], so lanes
# alternate across GPUs and each GPU ends up with JOBS_PER_GPU of them.
NLANES=$(( NGPU * JOBS_PER_GPU ))

JOBS=()
for seed in $SEEDS; do
    for arm in $ARMS; do
        JOBS+=("${arm}:${seed}")
    done
done

echo "Queue: ${#JOBS[@]} runs over $NLANES lanes ($NGPU GPUs x $JOBS_PER_GPU per GPU)"
for l in $(seq 0 $(( NLANES - 1 ))); do
    gpu=${GPU_ARR[$(( l % NGPU ))]}
    line="  lane$l (GPU${gpu}):"
    for i in "${!JOBS[@]}"; do
        [ $(( i % NLANES )) -eq "$l" ] && line="$line  ${JOBS[$i]/:/ s}"
    done
    echo "$line"
done

if [ "$DRY_RUN" != "0" ]; then
    echo "DRY_RUN=1 — nothing launched."
    exit 0
fi

mkdir -p paper/square/logs

# One background worker per lane, each walking its slice of the queue in order.
STAMP=$(date +%Y%m%d_%H%M%S)
for l in $(seq 0 $(( NLANES - 1 ))); do
    gpu=${GPU_ARR[$(( l % NGPU ))]}
    slice=()
    for i in "${!JOBS[@]}"; do
        [ $(( i % NLANES )) -eq "$l" ] && slice+=("${JOBS[$i]}")
    done
    [ ${#slice[@]} -eq 0 ] && continue
    worker_log="paper/square/logs/lane${l}_gpu${gpu}_${STAMP}.log"
    delay=$(( l * STAGGER ))
    (
        [ "$delay" -gt 0 ] && sleep "$delay"
        for job in "${slice[@]}"; do
            arm="${job%%:*}"; seed="${job##*:}"
            # Keep going if one arm dies so the rest of the sweep still runs.
            GPU="$gpu" paper/square/table1.sh "$arm" "$seed" \
                || echo "FAILED: $arm seed $seed on GPU$gpu"
        done
        echo "lane${l} finished"
    ) > "$worker_log" 2>&1 &
    echo "lane$l GPU${gpu} pid=$! start=+${delay}s log=${worker_log} jobs=${slice[*]}"
done

cat <<'EOF'

Launched. Follow along with:
    tail -f paper/square/logs/lane0_*.log           # which run is active
    tail -f paper/square/logs/vgp_square_seed_1_*.log   # that run's output
    nvidia-smi
Stop everything with:  pkill -f policy_robosuite/train.py
EOF
