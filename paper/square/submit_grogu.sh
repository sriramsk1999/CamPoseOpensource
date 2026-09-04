#!/usr/bin/env bash
# Submit Square Assembly runs on Grogu.
#
#   paper/square/submit_grogu.sh                    # table 1: 3 arms x seeds 1 2 3
#   paper/square/submit_grogu.sh table2             # table 2: 9 arms x 3 seeds
#   paper/square/submit_grogu.sh all                # all 12 arms x 3 seeds
#   paper/square/submit_grogu.sh vgp                # one arm, all seeds
#   SEEDS="1" paper/square/submit_grogu.sh          # one seed
#   DRY_RUN=1 paper/square/submit_grogu.sh          # print, submit nothing
#   ARMS="vgp plucker canonical dp act" paper/square/submit_grogu.sh
#
# Defaults to the 3090s: partition dheld (tier 5, 14-day cap), -C rtx3090.
# They live on grogu-4-8 and grogu-4-3, 16 GPUs total, and are the least
# contended thing on the cluster. A run is ~3 days on a 3090, so the 14-day cap
# means no CHAIN= dependency juggling is needed -- unlike ArticuBot's runs,
# which need it under deepaklong's 2-day cap.
#
#   PARTITION=dheld_dev GPU=rtx3090 TIME=1-00:00:00 paper/square/submit_grogu.sh
#
# Why 3090 and not something bigger: a 3090 is 24 GB, the same as the 4090 the
# arms were profiled on, so the measured footprints (14.5 GiB vgp / 16.5 GiB
# plucker) carry over unchanged and one job per GPU is right. A6000/A6000Ada at
# 48 GB would fit two per card, but only 8 of them are free against 15 3090s.
# Do NOT aim at 6000Blackwell (sm120) -- torch 2.5.1 ships no kernels for it.
set -euo pipefail

TABLE1=(vgp plucker canonical)
TABLE2=(dp spatial_softmax max_pool resnet18 dinov2_frozen dinov2_finetuned
        act vgp_sinusoidal rope3d)

# Accepts arm names, or the group keywords table1 / table2 / all.
ARMS_IN=()
for a in "$@"; do
    case "$a" in
        table1) ARMS_IN+=("${TABLE1[@]}") ;;
        table2) ARMS_IN+=("${TABLE2[@]}") ;;
        all)    ARMS_IN+=("${TABLE1[@]}" "${TABLE2[@]}") ;;
        *)      ARMS_IN+=("$a") ;;
    esac
done
if [ ${#ARMS_IN[@]} -eq 0 ]; then
    ARMS_IN=("${TABLE1[@]}")
fi
SEEDS="${SEEDS:-1 2 3}"
PARTITION="${PARTITION:-dheld}"
GPU="${GPU:-rtx3090}"
TIME="${TIME:-7-00:00:00}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p grogu_logs

n=0
for arm in "${ARMS_IN[@]}"; do
    case " ${TABLE1[*]} ${TABLE2[*]} " in
        *" $arm "*) ;;
        *) echo "no such arm: $arm" >&2; exit 1 ;;
    esac
    for seed in $SEEDS; do
        cmd=(sbatch -J "sq-${arm}-s${seed}" -p "$PARTITION" -C "$GPU" -t "$TIME"
             --export=ALL,ARM="${arm}",SEED="${seed}" paper/square/grogu.slurm)
        if [ "$DRY_RUN" != "0" ]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
        n=$((n + 1))
    done
done
echo "$n jobs ($([ "$DRY_RUN" != "0" ] && echo 'dry run' || echo submitted) on $PARTITION / $GPU)"
