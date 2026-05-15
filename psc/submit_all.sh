#!/bin/bash
# Convenience: submit all 6 baselines.
# Usage: bash psc/submit_all.sh    (from anywhere; we cd to the repo root)
#
# The slurm scripts use $SLURM_SUBMIT_DIR to find the repo, so we must
# sbatch from the repo root (not from psc/), so $SLURM_SUBMIT_DIR points
# at the repo root rather than at psc/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
sbatch psc/act_dino_sv.slurm
sbatch psc/dit_dino_sv_plucker.slurm
sbatch psc/dit_dino_sv_canon.slurm
sbatch psc/flow_matching_3dfa.slurm
sbatch psc/dit_maniwhere_sv.slurm
sbatch psc/dit_rope4d_dino_cv.slurm
echo "All 6 jobs submitted. Logs will land in psc/logs/ relative to $REPO_ROOT."
